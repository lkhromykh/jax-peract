import os
import logging

import jax
import optax
import cloudpickle
from flax import core
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from src.config import Config
from src.train_state import TrainState
from src.peract import PerAct
from src.rlbench_env.enviroment import RLBenchEnv
from src.rlbench_env.dataset import as_tfdataset
from src.behavior_cloning import bc, StepFn
import src.types_ as types

Params = core.FrozenDict[str, types.Array]


class Builder:

    CONFIG = 'config.yaml'
    STATE = 'state.cpkl'
    DEMO = 'demo'

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if not os.path.exists(path := self.exp_path()):
            logging.info('Init experiment dir.')
            os.makedirs(path)
        if not os.path.exists(path := self.exp_path(Builder.CONFIG)):
            cfg.save(path)

    def make_env(self, rng: types.RNG) -> RLBenchEnv:
        """training env ctor."""
        c = self.cfg
        max_int = jax.numpy.iinfo(jax.numpy.int32).max
        rng = jax.random.randint(rng, (), 1, max_int).item()
        scene_bounds = c.scene_lower_bound, c.scene_upper_bound
        return RLBenchEnv(seed=rng,
                          scene_bounds=scene_bounds,
                          scene_bins=c.scene_bins,
                          rot_bins=c.rot_bins,
                          time_limit=c.time_limit,
                          )

    def make_networks_and_params(
            self,
            seed: types.RNG,
            env: RLBenchEnv
    ) -> tuple[PerAct, Params]:
        nets = PerAct(self.cfg,
                      env.observation_spec(),
                      env.action_spec()
                      )
        obs = jax.tree_util.tree_map(lambda x: x.generate_value(),
                                     env.observation_spec())
        params = nets.init(seed, obs)
        nparams = sum(jax.tree_util.tree_map(
            lambda x: x.size, jax.tree_leaves(params)))
        logging.info(f'Number of params: {nparams}')
        return nets, params

    def make_state(self,
                   rng: types.RNG,
                   params: Params,
                   ) -> TrainState:
        if os.path.exists(path := self.exp_path(Builder.STATE)):
            logging.info('Loading existing state.')
            with open(path, 'rb') as f:
                state = cloudpickle.load(f)
            return jax.device_put(state)
        c = self.cfg
        optim = optax.lamb(c.learning_rate, weight_decay=c.weight_decay)
        clip = optax.clip_by_global_norm(c.max_grad_norm)
        optim = optax.chain(clip, optim)
        return TrainState.init(rng=rng,
                               params=params,
                               optim=optim,
                               )

    def make_dataset(self, env: RLBenchEnv) -> tf.data.Dataset:
        if os.path.exists(path := self.exp_path(Builder.DEMO)):
            logging.info('Loading existing dataset.')
            ds = tf.data.Dataset.load(path)
        else:
            logging.info('Collecting demos.')
            ds = env.get_demos(self.cfg.num_demos)
            ds = as_tfdataset(ds)
            ds.save(path)
        ds = ds.cache()\
           .repeat()\
           .shuffle(10 * self.cfg.batch_size)\
           .batch(self.cfg.batch_size)\
           .prefetch(tf.data.AUTOTUNE)
        return ds.as_numpy_iterator()

    def make_step_fn(self, nets: PerAct) -> StepFn:
        fn = bc(self.cfg, nets)
        if self.cfg.jit:
            fn = jax.jit(fn)
        return fn

    def exp_path(self, path: str = os.path.curdir) -> str:
        logdir = os.path.abspath(self.cfg.logdir)
        path = os.path.join(logdir, path)
        return os.path.abspath(path)
