from typing import Callable

import jax
import jax.numpy as jnp
import jmp
import optax

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
                loss_scale: jmp.LossScale,
                obs_t: types.Observation,
                act_t: types.Action,
                ) -> tuple[float | jnp.ndarray, types.Metrics]:
        policy_t = nets.apply(params, obs_t)
        loss = -policy_t.log_prob(act_t)
        ent_t = policy_t.entropy()
        return loss_scale.scale(loss), dict(loss=loss, entropy=ent_t)

    def step(state: TrainState, batch: types.Trajectory
             ) -> tuple[TrainState, types.Metrics]:
        params = state.params
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad_fn = jax.vmap(grad_fn, in_axes=(None, None, 0, 0))
        out = grad_fn(params, state.loss_scale,
                      batch['observations'], batch['actions']
                      )
        grad, metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), out)
        state = state.update(grad=grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state, metrics

    return step
