import os
import logging

import jax
import optax
import numpy as np
import cloudpickle
import tensorflow as tf
import flax.linen as nn
from flax import traverse_util
from jax.tree_util import tree_map, tree_leaves
tf.config.set_visible_devices([], 'GPU')

from src import utils
import src.types_ as types
from src.config import Config
from src.behavior_cloning import bc
from src.environment import RLBenchEnv
from src.networks.peract import PerAct
from src.dataset.dataset import DemosDataset
from src.train_state import TrainState, Params
from src.peract_env_wrapper import PerActEncoders, PerActEnvWrapper
from src.dataset.keyframes_extraction import extractor_factory


class Builder:
    """Construct all the required objects."""

    CONFIG = 'config.yaml'
    STATE = 'state.cpkl'

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if not os.path.exists(path := self.exp_path()):
            logging.info('Init experiment dir.')
            os.makedirs(path)
        if not os.path.exists(path := self.exp_path(Builder.CONFIG)):
            cfg.save(path)  # TODO: fix config
        np.random.seed(cfg.seed)
        tf.random.set_seed(cfg.seed)

    def make_env(self) -> PerActEnvWrapper:
        """Create and wrap an environment."""
        c = self.cfg
        encoders = PerActEncoders.from_config(c)
        env = RLBenchEnv(
            scene_bounds=c.scene_bounds,
            time_limit=c.time_limit,
        )
        return PerActEnvWrapper(
            env=env,
            encoders=encoders
        )

    def make_networks_and_params(self) -> tuple[PerAct, Params]:
        encoders = PerActEncoders.from_config(self.cfg)
        nets = PerAct(
            config=self.cfg,
            action_spec=encoders.action_spec()
        )
        obs = tree_map(lambda x: x.generate_value(), encoders.observation_spec())
        rng1, rng2 = jax.random.split(jax.random.PRNGKey(self.cfg.seed + 1))
        params = nets.init(rng1, obs)
        logging.info(nn.tabulate(nets, rng2)(obs))
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
        if os.path.exists(path := self.exp_path(Builder.STATE)):
            logging.info('Loading existing state.')
            state = self.load(path)
            return state
        optim = self.make_optim(params)
        rng = jax.random.PRNGKey(self.cfg.seed + 2)
        return TrainState.init(rng=rng,
                               params=params,
                               optim=optim,
                               )

    # TODO: tune pipeline
    def make_tfdataset(self) -> tf.data.Dataset:
        c = self.cfg
        enc = PerActEncoders.from_config(c)
        dds = DemosDataset(c.dataset_dir)
        extract_fn = extractor_factory(observation_transform=enc.infer_state,
                                       keyframe_transform=enc.infer_action)

        def as_trajectory_generator():
            for demo in dds.as_demo_generator():
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
        ds = ds.cache() \
             .repeat() \
             .map(utils.augmentations.select_random_transition) \
             .map(lambda item: utils.augmentations.voxel_grid_random_shift(item, c.max_shift)) \
             .batch(c.batch_size) \
             .prefetch(tf.data.AUTOTUNE)
        return ds

    def make_step_fn(self, nets: PerAct) -> types.StepFn:
        fn = bc(self.cfg, nets)
        if self.cfg.jit:
            fn = jax.jit(fn)
        return fn

    def exp_path(self, path: str = os.path.curdir) -> str:
        logdir = os.path.abspath(self.cfg.logdir)
        path = os.path.join(logdir, path)
        return os.path.abspath(path)

    def save(self, obj, path):
        with open(self.exp_path(path), 'wb') as f:
            cloudpickle.dump(obj, f)

    def load(self, path):
        with open(self.exp_path(path), 'rb') as f:
            return cloudpickle.load(f)
