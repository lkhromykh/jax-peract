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
    chex.assert_gpu_available()

    def loss_fn(params: Params,
                obs: types.Observation,
                act: types.Action,
                ) -> tuple[float | jnp.ndarray, types.Metrics]:
        chex.assert_rank(act, 1)
        policy = nets.apply(params, obs)
        cross_ent = -policy.log_prob(act)
        pos_ent = policy.distributions[0].entropy()
        low_dim_ent = policy.entropy() - pos_ent
        predict = policy.mode()
        (pos_pred, low_dim_pred), (pos_expert, low_dim_expert) = map(
            lambda x: jnp.split(x, [3]), (act, predict))
        return cross_ent, dict(
            cross_entropy=cross_ent,
            pos_entropy=pos_ent,
            low_dim_entropy=low_dim_ent,
            pos_accuracy=jnp.array_equal(pos_pred, pos_expert),
            low_dim_accuracy=jnp.array_equal(low_dim_pred, low_dim_expert)
        )

    @chex.assert_max_traces(1)
    def step(state: TrainState, batch: types.Trajectory
             ) -> tuple[TrainState, types.Metrics]:
        logging.info('Tracing BC step.')
        params = state.params
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad_fn = jax.vmap(grad_fn, in_axes=(None, 0, 0))
        out = grad_fn(params, batch['observations'], batch['actions'])
        grad, metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), out)
        state = state.update(grad=grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state, metrics

    return step
