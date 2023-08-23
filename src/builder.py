from typing import Any

import jax
import jax.numpy as jnp
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
        return Networks(self.cfg, 10)

    def make_state(self, rng: jax.random.PRNGKey,
                   params: core.FrozenDict[str, Any],
                   optim: optax.TransformUpdateFn
                   ) -> TrainState:
        return TrainState.init(rng=rng,
                               params=params,
                               optim=optim,
                               target_update_var=0)

    def make_optim(self) -> optax.TransformUpdateFn:
        optim = optax.lamb(self.cfg.learning_rate)
        clip = optax.clip_by_global_norm(self.cfg.max_grad_norm)
        return optax.chain(clip, optim)

    def make_dataset(self, split):
        return ops.get_dataset(split, batch_size=self.cfg.batch_size,
                               img_size=self.cfg.img_size)

    def make_step_fn(self, nets: Networks):
        step = ops.supervised(self.cfg, nets)
        if self.cfg.jit:
            step = jax.jit(step)
        return step
