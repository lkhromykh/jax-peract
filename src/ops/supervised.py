import jax
import jax.numpy as jnp
import optax
import chex

from src.train_state import TrainState
from src.networks import Perceiver
from src.config import Config
from src import types_ as types
from src.ops.augmentations import augmentation_fn


def supervised(cfg: Config, nets: Perceiver):

    def loss_fn(params, rng, img, label):
        img = augmentation_fn(rng, img, 2)
        logits = nets.apply(params, img)
        acc = jnp.mean(logits.argmax(-1) == label.argmax(-1))
        loss = optax.softmax_cross_entropy(logits, label).mean()
        return loss, dict(accuracy=acc, loss=loss)

    @chex.assert_max_traces(n=1)
    def step(state: TrainState, batch) -> tuple[TrainState, types.Metrics]:
        params = state.params
        rng, subkey = jax.random.split(state.rng)
        imgs, labels = batch
        grad, metrics = jax.grad(loss_fn, has_aux=True)(params, subkey, imgs, labels)
        state = state.update(grad=grad)
        metrics.update(grad_norm=optax.global_norm(grad))
        return state.replace(rng=rng), metrics

    return step
