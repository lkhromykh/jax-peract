from typing import Any

import dm_env
import jax
import optax
from flax import core

from src.config import Config
from src.train_state import TrainState
from src.networks import Networks
from src.rlbench.enviroment import RLBenchEnv
from src.behavior_cloning import bc, StepFn
import src.types_ as types


class Builder:

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def make_networks(self) -> Networks:
        return Networks(self.cfg)

    def make_env(self) -> RLBenchEnv:
        """training env ctor."""
        ...

    def make_state(self,
                   rng: jax.random.PRNGKey,
                   params: core.FrozenDict[str, Any],
                   optim: optax.GradientTransformation
                   ) -> TrainState:
        return TrainState.init(rng=rng,
                               params=params,
                               optim=optim)

    def make_optim(self) -> optax.GradientTransformation:
        c = self.cfg
        optim = optax.lamb(c.learning_rate, weight_decay=c.weight_decay)
        clip = optax.clip_by_global_norm(c.max_grad_norm)
        return optax.chain(clip, optim)

    def make_step_fn(self, nets: Networks) -> StepFn:
        fn = bc(self.cfg, nets)
        if self.cfg.jit:
            fn = jax.jit(fn)
        return fn
