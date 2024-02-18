import pathlib

import jax
import optax
import numpy as np
import cloudpickle
import tensorflow as tf
import flax.linen as nn
from flax import traverse_util
from jax.tree_util import tree_map
tf.config.set_visible_devices([], 'GPU')

from src import utils
import src.types_ as types
from src.config import Config
from src.behavior_cloning import bc
from src.environment import RLBenchEnv, GoalConditionedEnv
from src.networks.peract import PerAct
from src.dataset.dataset import DemosDataset
from src.train_state import TrainState, Params
from src.logger import get_logger, logger_add_file_handler
from src.dataset.keyframes_extraction import extractor_factory
from src.peract_env_wrapper import PerActEncoders, PerActEnvWrapper


class Builder:
    """Construct all the required objects."""

    CONFIG = 'config.yaml'
    STATE = 'state.cpkl'
    LOGS = 'logs'
    DATASETS_DIR = 'processed_demos/'

    def __init__(self, cfg: Config) -> None:
        """Prepare an experiment space."""
        self.cfg = cfg
        self.exp_path().mkdir(parents=True, exist_ok=True)
        if not (path := self.exp_path(Builder.CONFIG)).exists():
            cfg.save(path)
        logger_add_file_handler(self.exp_path(Builder.LOGS))

    def make_encoders(self) -> PerActEncoders:
        """Create transformations required to infer state and action from observation."""
        c = self.cfg
        scene_encoder = utils.VoxelGrid(
            scene_bounds=c.scene_bounds,
            nbins=c.scene_bins
        )
        action_encoder = utils.DiscreteActionTransform(
            scene_bounds=c.scene_bounds,
            scene_bins=c.scene_bins,
            rot_bins=c.rot_bins,
        )
        text_encoder = utils.CLIP(max_length=c.text_context_length)
        return PerActEncoders(
            scene_encoder=scene_encoder,
            action_encoder=action_encoder,
            text_encoder=text_encoder
        )

    def make_env(self, encoders: PerActEncoders | None = None) -> PerActEnvWrapper | GoalConditionedEnv:
        """Create and wrap an environment."""
        c = self.cfg
        env = RLBenchEnv(
            scene_bounds=c.scene_bounds,
            time_limit=c.time_limit,
        )
        if encoders is None:
            return env
        return PerActEnvWrapper(
            env=env,
            encoders=encoders
        )

    def make_networks_and_params(self, encoders: PerActEncoders) -> tuple[PerAct, Params]:
        nets = PerAct(
            config=self.cfg,
            action_spec=encoders.action_spec()
        )
        obs = tree_map(lambda x: x.generate_value(), encoders.observation_spec())
        rng1, rng2 = jax.random.split(jax.random.PRNGKey(self.cfg.seed + 1))
        params = nets.init(rng1, obs)
        get_logger().info(nn.tabulate(nets, rng2)(obs))
        return nets, params

    def make_optim(self, params: Params) -> optax.GradientTransformation:
        c = self.cfg
        if c.warmup_steps > 0:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.,
                peak_value=c.peak_learning_rate,
                warmup_steps=c.warmup_steps,
                decay_steps=c.training_steps
            )
        else:
            schedule = optax.constant_schedule(c.peak_learning_rate)
        mask = traverse_util.path_aware_map(
            lambda path, _: path[-1] not in ('bias', 'scale'),
            params
        )
        mask = type(params)(mask)
        return optax.chain(
            optax.clip_by_global_norm(c.max_grad_norm),
            optax.scale_by_adam(),
            optax.add_decayed_weights(c.weight_decay, mask),
            optax.scale_by_trust_ratio(),
            optax.scale_by_schedule(schedule),
            optax.scale(-1)
        )

    def make_state(self, params: Params) -> TrainState:
        if (path := self.exp_path(Builder.STATE)).exists():
            get_logger().info('Loading existing state.')
            state = self.load(path)
            return state
        optim = self.make_optim(params)
        rng = jax.random.PRNGKey(self.cfg.seed + 2)
        return TrainState.init(rng=rng,
                               params=params,
                               optim=optim,
                               )

    def parse_demos(self, dataset_dir: str | pathlib.Path, save_only: bool = True) -> tf.data.Dataset | None:
        dataset_dir = pathlib.Path(dataset_dir).resolve()
        dataset = DemosDataset(dataset_dir)
        enc = self.make_encoders()
        extract_fn = extractor_factory(observation_transform=enc.infer_state,
                                       keyframe_transform=enc.infer_action)

        def as_trajectory_generator():
            for demo in dataset.as_demo_generator():
                pairs, _ = extract_fn(demo)
                def nested_stack(ts): return tree_map(lambda *xs: np.stack(xs), *ts)
                obs, act = map(nested_stack, zip(*pairs))
                yield types.Trajectory(observations=obs, actions=act)

        def to_tf_specs(x):
            return tf.TensorSpec(shape=(None,) + x.shape,
                                 dtype=tf.as_dtype(x.dtype))

        output_signature = types.Trajectory(
            observations=tree_map(to_tf_specs, enc.observation_spec()),
            actions=tf.TensorSpec(shape=(None, len(enc.action_spec())), dtype=tf.int32)
        )
        ds = tf.data.Dataset.from_generator(
            generator=as_trajectory_generator,
            output_signature=output_signature
        )
        local_ds_path = self.exp_path(Builder.DATASETS_DIR) / dataset_dir.name
        get_logger().info('Saving to %s', local_ds_path)
        if save_only:
            return ds.save(str(local_ds_path), compression='GZIP')
        return ds

    def make_tfdataset(self) -> tf.data.Dataset:
        processed_ds_path = self.exp_path(Builder.DATASETS_DIR)
        if not processed_ds_path.exists():
            import multiprocessing as mp

            demos_path = pathlib.Path(self.cfg.datasets_dir)
            assert demos_path.exists()
            get_logger().info('Parsing datasets.')
            processed_ds_path.mkdir(parents=False)
            dataset_paths = list(demos_path.glob('[!.]*/'))
            with mp.Pool(processes=len(dataset_paths)) as pool:
                pool.map(self.parse_demos, dataset_paths)
        c = self.cfg
        tf.random.set_seed(c.seed)
        np.random.seed(c.seed)
        action_encoder = self.make_encoders().action_encoder
        datasets = [tf.data.Dataset.load(str(p), compression='GZIP') for p in processed_ds_path.iterdir()]
        ds = tf.data.Dataset.sample_from_datasets(datasets,
                                                  stop_on_empty_dataset=False,
                                                  rerandomize_each_iteration=True)
        # ds = ds.interleave(utils.augmentations.select_random_transition,
        #                    cycle_length=1,
        #                    num_parallel_calls=tf.data.AUTOTUNE) \
        # cache, shuffle, interleave?
        ds = ds.repeat() \
               .map(utils.augmentations.select_random_transition,
                    num_parallel_calls=tf.data.AUTOTUNE) \
               .map(lambda item: utils.augmentations.scene_rotation(item, action_encoder)) \
               .map(lambda item: utils.augmentations.scene_shift(item, c.max_shift),
                    num_parallel_calls=tf.data.AUTOTUNE) \
               .batch(c.batch_size,
                      num_parallel_calls=tf.data.AUTOTUNE,
                      drop_remainder=True) \
               .prefetch(tf.data.AUTOTUNE)
        return ds

    def make_step_fn(self, nets: PerAct) -> types.StepFn:
        fn = bc(self.cfg, nets)
        if self.cfg.jit:
            fn = jax.jit(fn)
        return fn

    def exp_path(self, path: str | pathlib.Path = pathlib.Path()) -> pathlib.Path:
        logdir = pathlib.Path(self.cfg.logdir)
        return logdir.joinpath(path).resolve()

    def save(self, obj, path):
        with open(self.exp_path(path), 'wb') as f:
            cloudpickle.dump(obj, f)

    def load(self, path):
        with open(self.exp_path(path), 'rb') as f:
            return cloudpickle.load(f)
