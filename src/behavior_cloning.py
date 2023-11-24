import logging
from typing import Callable

import jax
import jax.numpy as jnp
import optax
import chex

from src.config import Config
from src.peract import PerAct
from src.train_state import TrainState, Params
from src import types_ as types


StepFn = Callable[
    [TrainState, types.Trajectory],
    tuple[TrainState, types.Metrics]
]


def bc(cfg: Config, nets: PerAct) -> StepFn:

    def loss_fn(params: Params,
                obs: types.Observation,
                act: types.Action,
                ) -> tuple[float | jnp.ndarray, types.Metrics]:
        policy = nets.apply(params, obs)
        cross_ent = -policy.log_prob(act)
        ent = policy.entropy()
        return cross_ent - cfg.ent_coef * ent, dict(cross_entropy=cross_ent,
                                                    entropy=ent)

    @chex.assert_max_traces(1)
    def step(state: TrainState, batch: types.Trajectory
             ) -> tuple[TrainState, types.Metrics]:
        logging.info('Tracing BC step.')
        params = state.params
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad_fn = jax.vmap(grad_fn, in_axes=(None, 0, 0))
        out = grad_fn(params,
                      batch['observations'], batch['actions']
                      )
        grad, metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), out)
        state = state.update(grad=grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state, metrics

    return step
