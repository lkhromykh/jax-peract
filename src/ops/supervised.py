import jax
import jax.numpy as jnp
import optax
import chex

from src.train_state import TrainState
from src.networks import Networks
from src.config import Config
from src import types_ as types


def supervised(cfg: Config, nets: Networks):

    def loss_fn(params, img, label):
        logits = nets.apply(params, img)
        acc = jnp.mean(logits.argmax(-1) == label)
        conf = jax.nn.softmax(logits, -1).max(-1).mean()
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()
        return loss, dict(accuracy=acc, loss=loss, confidence=conf)

    @chex.assert_max_traces(n=1)
    def step(state: TrainState, batch) -> tuple[TrainState, types.Metrics]:
        params = state.params
        imgs, labels = batch
        grad, metrics = jax.grad(loss_fn, has_aux=True)(params, imgs, labels)
        state = state.update(grad=grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state, metrics

    return step
