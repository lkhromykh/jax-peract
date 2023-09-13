from typing import Callable

import jax
import jax.numpy as jnp
import optax

from src.config import Config
from src.networks import Networks
from src.train_state import TrainState, Params
from src import types_ as types


StepFn = Callable[
    [TrainState, types.Trajectory],
    tuple[TrainState, types.Metrics]
]


def bc(cfg: Config, nets: Networks) -> StepFn:

    def loss_fn(params: Params,
                obs_t: types.Observation,
                act_t: types.Action
                ) -> tuple[float | jnp.ndarray, types.Metrics]:
        import pdb; pdb.set_trace()
        policy_t = nets.apply(params, obs_t)
        log_prob_t = policy_t.log_prob(act_t)
        loss = -log_prob_t.mean()
        return loss, dict(loss=loss, entropy=policy_t)

    def step(state: TrainState, batch: types.Trajectory
             ) -> tuple[TrainState, types.Metrics]:
        params = state.params
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad, metrics = grad_fn(params, batch['observations'], batch['actions'])
        state = state.update(grad=grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state, metrics

    return step
