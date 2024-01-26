import os
import logging

import jax
import optax
import cloudpickle
from flax import traverse_util
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from src.config import Config
from src.train_state import TrainState, Params
from src.peract import PerAct
from src.rlbench_env.enviroment import RLBenchEnv
from src.rlbench_env.dataset import as_tfdataset
from src.behavior_cloning import bc, StepFn
from src.utils.augmentations import random_shift
import src.types_ as types


class Builder:

    CONFIG = 'config.yaml'
    SPECS = 'specs.cpkl'
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
        return RLBenchEnv(
            seed=rng,
            scene_bounds=c.scene_bounds,
            scene_bins=c.scene_bins,
            rot_bins=c.rot_bins,
            text_emb_length=c.text_emb_len,
            time_limit=c.time_limit,
        )

    def make_networks_and_params(
            self,
            rng: types.RNG,
            env_specs: types.EnvSpecs
    ) -> tuple[PerAct, Params]:
        obs_spec, act_spec = env_specs
        nets = PerAct(
            config=self.cfg,
            observation_spec=obs_spec,
            action_spec=act_spec
        )
        obs = jax.tree_util.tree_map(lambda x: x.generate_value(), obs_spec)
        params = nets.init(rng, obs)
        nparams = sum(jax.tree_util.tree_map(
            lambda x: x.size, jax.tree_leaves(params)))
        logging.info(f'Number of params: {nparams}')
        return nets, params

    def make_optim(self, params: Params) -> optax.GradientTransformation:
        c = self.cfg
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.,
            peak_value=c.peak_learning_rate,
            warmup_steps=c.warmup_steps,
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

    def make_state(
            self,
            rng: types.RNG,
            params: Params,
    ) -> TrainState:
        if os.path.exists(path := self.exp_path(Builder.STATE)):
            logging.info('Loading existing state.')
            state = self.load(path)
            return jax.device_put(state)
        optim = self.make_optim(params)
        return TrainState.init(rng=rng,
                               params=params,
                               optim=optim,
                               )

    def make_dataset_and_specs(
            self,
            rng: types.RNG,
            env: RLBenchEnv | None
    ) -> tuple[tf.data.Dataset, types.EnvSpecs]:
        c = self.cfg
        if os.path.exists(path := self.exp_path(Builder.DEMO)):
            logging.info('Loading existing dataset and specs.')
            ds = tf.data.Dataset.load(path)
            specs = self.load(Builder.SPECS)
        elif env is not None:
            logging.info('Collecting demos.')
            ds = env.get_demos(c.num_demos)
            ds = as_tfdataset(ds)
            specs = (env.observation_spec(), env.action_spec())
            ds.save(path)
            self.save(specs, Builder.SPECS)
        else:
            raise RuntimeError('Impossible to obtain demos.')
        max_int = jax.numpy.iinfo(jax.numpy.int32).max
        rng = jax.random.randint(rng, (), 1, max_int).item()
        tf.random.set_seed(rng)
        ds = ds.cache()\
               .repeat()\
               .map(lambda x: random_shift(x, c.max_shift)) \
               .shuffle(100 * c.batch_size) \
               .batch(c.batch_size)\
               .prefetch(tf.data.AUTOTUNE)
        return ds.as_numpy_iterator(), specs

    def make_step_fn(self, nets: PerAct) -> StepFn:
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
