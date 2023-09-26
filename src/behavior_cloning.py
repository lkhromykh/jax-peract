from typing import Callable

import jax
import jax.numpy as jnp
import jmp
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
    # amp = jmp.get_policy(cfg.precision)

    def loss_fn(params: Params,
                loss_scale: jmp.LossScale,
                obs_t: types.Observation,
                act_t: types.Action,
                ) -> tuple[float | jnp.ndarray, types.Metrics]:
        policy_t = nets.apply(params, obs_t)
        ent_t = policy_t.entropy()
        log_prob_t = policy_t.log_prob(act_t)
        loss = -log_prob_t.mean()
        return loss_scale.scale(loss), dict(loss=loss, entropy=ent_t.mean())

    def step(state: TrainState, batch: types.Trajectory
             ) -> tuple[TrainState, types.Metrics]:
        params = state.params
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad, metrics = grad_fn(params, state.loss_scale,
                                batch['observations'], batch['actions']
                                )
        state = state.update(grad=grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state, metrics

    return step
