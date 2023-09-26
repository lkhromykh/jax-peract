from typing import Any

import jax
import jmp
import optax
from flax import core

from src.config import Config
from src.train_state import TrainState
from src.networks import Networks
from src.rlbench_env.enviroment import RLBenchEnv
from src.rlbench_env.action_mode import ActionMode
from src.behavior_cloning import bc, StepFn


class Builder:

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def make_networks(self, env: RLBenchEnv) -> Networks:
        return Networks(self.cfg,
                        env.observation_spec(),
                        env.action_spec()
                        )

    def make_env(self, rng: int) -> RLBenchEnv:
        """training env ctor."""
        c = self.cfg
        scene_bounds = c.scene_lower_bound, c.scene_upper_bound
        nbins = ActionMode.Discretization(c.scene_nbins,
                                          c.rot_nbins,
                                          c.grip_nbins)
        return RLBenchEnv(rng=rng,
                          scene_bounds=scene_bounds,
                          nbins=nbins,
                          time_limit=c.time_limit
                          )

    def make_state(self,
                   rng: jax.random.PRNGKey,
                   params: core.FrozenDict[str, Any],
                   ) -> TrainState:
        optim = self.make_optim()
        return TrainState.init(rng=rng,
                               params=params,
                               optim=optim,
                               loss_scale=jmp.NoOpLossScale()
                               )

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
