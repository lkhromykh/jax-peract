import logging
from typing import Callable

import jax
import jax.numpy as jnp
import optax
import chex
from flax import traverse_util

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
                observation: types.Observation,
                action: types.Action,
                ) -> tuple[float | jnp.ndarray, types.Metrics]:
        chex.assert_rank(action, 1)
        policy = nets.apply(params, observation)
        cross_ent = -policy.log_prob(action)
        metrics = dict(loss=cross_ent)
        # everything else is devoted to metrics computation.
        idx = 0
        dists_names = ['pos'] + [f'euler{i}' for i in range(3)] + ['grasp']
        for name, dist in zip(dists_names, policy.distributions):
            act_pred = jnp.atleast_1d(dist.mode())
            next_idx = idx + act_pred.size
            act_truth = action[idx:next_idx]
            def _join(metric): return f'{metric}_{name}'
            metrics |= {_join('entropy'): dist.entropy(),
                        _join('accuracy'): jnp.array_equal(act_truth, act_pred)}
            idx = next_idx
        return cross_ent, metrics

    @chex.assert_max_traces(1)
    def step(state: TrainState,
             batch: types.Trajectory
             ) -> tuple[TrainState, types.Metrics]:
        logging.info('Tracing BC step.')  # log params
        params = state.params
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad_fn = jax.vmap(grad_fn, in_axes=(None, 0, 0))
        out = grad_fn(params, batch['observations'], batch['actions'])
        grad, metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), out)
        layers_grad = traverse_util.flatten_dict(grad)
        layers_grad = {'grads_' + '_'.join(key): jnp.ravel(val)
                       for key, val in layers_grad.items()}
        layers_val = traverse_util.flatten_dict(params)
        layers_val = {'_'.join(key): jnp.ravel(val)
                      for key, val in layers_val.items()}
        metrics.update(layers_grad)
        metrics.update(layers_val)
        metrics.update(grad_norm=optax.global_norm(grad))
        state = state.update(grad=grad)
        return state, metrics

    return step
