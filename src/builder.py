import os
from typing import Any

import jax
import jmp
import optax
from flax import core

from src.config import Config
from src.train_state import TrainState
from src.networks import Networks
from src.rlbench_env.enviroment import RLBenchEnv
from src.behavior_cloning import bc, StepFn
import src.types_ as types


class Builder:

    CONFIG = 'config.yaml'
    STATE = 'state.cpkl'
    DEMOS = 'replay/'

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if not os.path.exists(path := self.exp_path(Builder.DEMOS)):
            os.makedirs(path)

    def make_env(self, rng: types.RNG) -> RLBenchEnv:
        """training env ctor."""
        c = self.cfg
        rng = rng[0].item()
        scene_bounds = c.scene_lower_bound, c.scene_upper_bound
        return RLBenchEnv(rng=rng,
                          scene_bounds=scene_bounds,
                          time_limit=c.time_limit,
                          )

    def make_networks_and_params(
            self,
            seed: types.RNG,
            env: RLBenchEnv
    ) -> tuple[Networks, core.FrozenDict[str, types.Array]]:
        nets = Networks(self.cfg,
                        env.observation_spec(),
                        env.action_spec()
                        )
        obs = jax.tree_util.tree_map(lambda x: x.generate_value(),
                                     env.observation_spec())
        params = nets.init(seed, obs)
        return nets, params

    def make_state(self,
                   rng: types.RNG,
                   params: core.FrozenDict[str, Any],
                   ) -> TrainState:
        c = self.cfg
        optim = optax.adamw(c.learning_rate, weight_decay=c.weight_decay)
        clip = optax.clip_by_global_norm(c.max_grad_norm)
        optim = optax.chain(clip, optim)
        return TrainState.init(rng=rng,
                               params=params,
                               optim=optim,
                               loss_scale=jmp.NoOpLossScale()
                               )

    def make_step_fn(self, nets: Networks) -> StepFn:
        fn = bc(self.cfg, nets)
        if self.cfg.jit:
            fn = jax.jit(fn)
        return fn

    def exp_path(self, path: str = os.path.curdir) -> str:
        logdir = os.path.abspath(self.cfg.logdir)
        path = os.path.join(logdir, path)
        return os.path.abspath(path)
