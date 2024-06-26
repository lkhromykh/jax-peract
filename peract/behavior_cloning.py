import jax
import jax.numpy as jnp
import optax
import chex

from peract.config import Config
from peract.logger import get_logger
from peract.networks.peract import PerAct
from peract.utils.distributions import Blockwise
from peract.train_state import TrainState, Params
from peract import types_ as types


def train(cfg: Config, nets: PerAct) -> types.StepFn:
    chex.assert_gpu_available()

    def loss_fn(params: Params,
                observation: types.State,
                action: types.Action,
                ) -> tuple[float | jnp.ndarray, types.Metrics]:
        chex.assert_rank(action, 1)
        policy = nets.apply(params, observation)
        metrics = _get_policy_metrics(policy, action)
        return metrics['cross_entropy'], metrics

    @chex.assert_max_traces(1)
    def step(state: TrainState,
             batch: types.Trajectory
             ) -> tuple[TrainState, types.Metrics]:
        get_logger().info('Tracing training step.')
        params = state.params
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grad_fn = jax.vmap(grad_fn, in_axes=(None, 0, 0))
        out = grad_fn(params, batch.observations, batch.actions)
        grad, metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), out)
        state = state.update(grad=grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state, metrics

    return step


def validate(cfg: Config, nets: PerAct) -> types.StepFn:

    def eval_fn(params: Params,
                observation: types.State,
                action: types.Action
                ) -> types.Metrics:
        chex.assert_rank(action, 1)
        policy = nets.apply(params, observation)
        metrics = _get_policy_metrics(policy, action)
        del metrics['pos_logits']
        return metrics

    @chex.assert_max_traces(2)  # drop_remainder=False
    def step(state: TrainState,
             batch: types.Trajectory
             ) -> tuple[TrainState, types.Metrics]:
        get_logger().info('Tracing validation step.')
        eval_ = jax.vmap(eval_fn, in_axes=(None, 0, 0))
        metrics = eval_(state.params, batch.observations, batch.actions)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)
        return tuple(map(jax.lax.stop_gradient, (state, metrics)))

    return step


def _get_policy_metrics(policy: Blockwise, expert_action: types.Action) -> types.Metrics:
    pos_dist, *low_dim_dists = policy.distributions
    metrics = dict(cross_entropy=-policy.log_prob(expert_action),
                   pos_logits=pos_dist.distribution.logits)

    def per_dist_metrics(dist_, labels, topk, postfix):
        metrics_ = {}
        argsorted_logits = jnp.argsort(dist_.logits)
        for k in topk:
            pred_labels = argsorted_logits[-k:]
            correct = jnp.isin(pred_labels, labels).any().astype(jnp.float32)
            metrics_[f'top_{k}_acc_{postfix}'] = correct
        metrics_[f'cross_entropy_{postfix}'] = -dist_.log_prob(labels)
        return metrics_

    metrics.update(per_dist_metrics(
        pos_dist.distribution,
        pos_dist.bijector.inverse(expert_action[:3]),
        topk=(1, 7),
        postfix='pos'
    ))
    components_names = ('yaw', 'pitch', 'roll', 'grasp', 'termsig')
    for name, dist, label in zip(components_names, low_dim_dists, expert_action[3:]):
        topk_ = (1, 3) if name in components_names[:3] else (1,)
        metrics |= per_dist_metrics(dist, label, topk=topk_, postfix=name)
    return metrics
