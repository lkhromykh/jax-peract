import pathlib
from typing import Literal

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
from src import behavior_cloning
from src.environment import RLBenchEnv, GoalConditionedEnv, UREnv
from src.networks.peract import PerAct
from src.train_state import TrainState, Params
from src.logger import get_logger, logger_add_file_handler
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
        if (path := self.exp_path(Builder.CONFIG)).exists():
            saved_cfg = Config.load(path)
            if saved_cfg != cfg:
                get_logger().warning('Warning! Config differs from the saved one.')
        else:
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

    def make_env(self,
                 encoders: PerActEncoders | None = None
                 ) -> PerActEnvWrapper | GoalConditionedEnv:
        """Create and wrap an environment."""
        c = self.cfg
        env = RLBenchEnv(
            scene_bounds=c.scene_bounds,
            time_limit=c.time_limit,
        )
       # env = UREnv(
       #     address=('192.168.1.136', 5555),
       #     scene_bounds=c.scene_bounds,
       #     time_limit=c.time_limit
       # )
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
            schedule = optax.cosine_decay_schedule(
                init_value=c.peak_learning_rate,
                decay_steps=c.training_steps
            )
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

    def make_tfdataset(self,
                       split: Literal['train', 'val']
                       ) -> tf.data.Dataset:
        c = self.cfg
        tf.random.set_seed(c.seed)
        np.random.seed(c.seed)
        action_encoder = self.make_encoders().action_encoder
        processed_ds_path = self.exp_path(Builder.DATASETS_DIR).resolve()

        def load_dataset(path):
            _ds = tf.data.Dataset.load(str(path))
            val_eps = max(int(c.val_split * len(_ds)), 1)
            match split:
                case 'val': _ds = _ds.take(val_eps)
                case 'train': _ds = _ds.skip(val_eps)
                case _: raise ValueError(split)
            _ds = _ds.flat_map(tf.data.Dataset.from_tensor_slices)
            if split == 'train':
                _ds = _ds.cache().repeat().shuffle(1000)  # assuming one episode has ~100 steps.
            return _ds.prefetch(tf.data.AUTOTUNE)

        datasets = [load_dataset(path) for path in sorted(processed_ds_path.iterdir())]
        ds = tf.data.Dataset.from_tensor_slices(datasets).interleave(lambda x: x)
        match split:
            case 'val':
                ds = ds.batch(2 * c.batch_size,
                              num_parallel_calls=tf.data.AUTOTUNE,
                              drop_remainder=False
                              ) \
                       .cache() \
                       .prefetch(tf.data.AUTOTUNE)
            case 'train':
                max_shift = int(c.max_trans_aug * c.scene_bins)
                ds = ds.map(lambda item: utils.augmentations.scene_rotation(item, action_encoder),
                            num_parallel_calls=tf.data.AUTOTUNE) \
                       .map(lambda item: utils.augmentations.scene_shift(item, max_shift),
                            num_parallel_calls=tf.data.AUTOTUNE) \
                       .batch(c.batch_size,
                              num_parallel_calls=tf.data.AUTOTUNE,
                              drop_remainder=True) \
                       .map(utils.augmentations.color_transforms,
                            num_parallel_calls=tf.data.AUTOTUNE) \
                       .prefetch(4)
            case _:
                raise ValueError(split)
        return ds

    def make_step_fn(self,
                     nets: PerAct,
                     step: Literal['train', 'val']
                     ) -> types.StepFn:
        match step:
            case 'val': fn = behavior_cloning.validate
            case 'train': fn = behavior_cloning.train
            case _: raise ValueError(step)
        fn = fn(self.cfg, nets)
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
