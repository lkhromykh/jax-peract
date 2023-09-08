from typing import Any

import jax
import optax
from flax import core

from src.config import Config
from src.train_state import TrainState
from src.networks import Networks
from src import ops


class Builder:

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def make_networks(self) -> Networks:
        return Networks(self.cfg)

    def make_env(self) -> 'dm_env.Environment':
        """training env ctor."""

    def make_state(self, rng: jax.random.PRNGKey,
                   params: core.FrozenDict[str, Any],
                   optim: optax.TransformUpdateFn
                   ) -> TrainState:
        return TrainState.init(rng=rng,
                               params=params,
                               optim=optim,
                               target_update_var=0)

    def make_optim(self) -> optax.TransformUpdateFn:
        c = self.cfg
        optim = optax.lamb(c.learning_rate, weight_decay=c.weight_decay)
        clip = optax.clip_by_global_norm(c.max_grad_norm)
        return optax.chain(clip, optim)

    def make_dataset(self, split):
        c = self.cfg
        return ops.get_dataset(split,
                               batch_size=c.batch_size,
                               img_size=c.img_size,
                               mixup_lambda=c.mixup_lambda)

    def make_step_fn(self, nets: Networks):
        step = ops.supervised(self.cfg, nets)
        if self.cfg.jit:
            step = jax.jit(step)
        return step
