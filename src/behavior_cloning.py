import jax
import jax.numpy as jnp
import optax
import chex

from src.config import Config
from src.logger import get_logger
from src.networks.peract import PerAct
from src.utils.distributions import Blockwise
from src.train_state import TrainState, Params
from src import types_ as types


def _get_policy_metrics(policy: Blockwise, expert_action: types.Action) -> types.Metrics:

    def per_dist_metrics(dist_, labels, topk=(1, 5), postfix=''):
        metrics_ = {}
        argsorted_logits = jnp.argsort(dist_.logits)
        for k in topk:
            pred_labels = argsorted_logits[-k:]
            correct = jnp.isin(pred_labels, labels).any().astype(jnp.float32)
            metrics_[f'top_{k}_acc_{postfix}'] = correct
        metrics_[f'entropy_{postfix}'] = dist_.entropy()
        return metrics_

    pos_dist, *low_dim_dists = policy.distributions
    metrics = per_dist_metrics(
        pos_dist.distribution,
        pos_dist.bijector.inverse(expert_action[:3]),
        topk=(1, 7),
        postfix='pos'
    )
    components_names = ('yaw', 'pitch', 'roll', 'grasp', 'termsig')
    for name, dist, label in zip(components_names, low_dim_dists, expert_action[3:]):
        topk_ = (1, 3) if name in components_names[:3] else (1,)
        metrics |= per_dist_metrics(dist, label, topk=topk_, postfix=name)
    return metrics


def train(cfg: Config, nets: PerAct) -> types.StepFn:
    chex.assert_gpu_available()

    def loss_fn(params: Params,
                observation: types.State,
                action: types.Action,
                ) -> tuple[float | jnp.ndarray, types.Metrics]:
        chex.assert_rank(action, 1)
        policy = nets.apply(params, observation)
        cross_ent = -policy.log_prob(action)
        metrics = dict(loss=cross_ent)
        metrics.update(_get_policy_metrics(policy, action))
        return cross_ent, metrics

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
        return _get_policy_metrics(policy, action)

    @chex.assert_max_traces(2)  # drop_remainder=False
    def step(state: TrainState,
             batch: types.Trajectory
             ) -> tuple[TrainState, types.Metrics]:
        get_logger().info('Tracing validation step.')
        eval_ = jax.vmap(eval_fn, in_axes=(None, 0, 0))
        metrics = eval_(state.params, batch.observations, batch.actions)
        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)
        return state, metrics

    return step
