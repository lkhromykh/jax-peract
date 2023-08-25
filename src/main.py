import jax
import jax.numpy as jnp
# jax.config.update('jax_platform_name', 'cpu')

from src.config import Config
from src.builder import Builder
from src import ops


def main():
    cfg = Config()
    builder = Builder(cfg)
    nets = builder.make_networks()
    train_ds = builder.make_dataset(split='train')
    step = builder.make_step_fn(nets)
    img, _ = next(train_ds)
    rng = jax.random.PRNGKey(cfg.seed)
    params = nets.init(rng, img)
    print('Number of params: ', ops.tree_size(params))
    optim = builder.make_optim()
    state = builder.make_state(rng, params, optim)

    def evaluate():
        predicts = []
        labels = []
        for img, label in builder.make_dataset(split='test[:10%]'):
            img = jax.device_put(img)
            logits = jax.jit(nets.apply)(state.params, img)
            predicts.append(logits.argmax(-1))
            labels.append(label.argmax(-1))
        predicts = jnp.concatenate(predicts)
        labels = jnp.concatenate(labels)
        return jnp.mean(predicts == labels)

    for t in range(cfg.training_steps):
        batch = jax.device_put(next(train_ds))
        state, metrics = step(state, batch)
        if t % cfg.eval_every == 0:
            metrics.update(step=t, eval_accuracy=evaluate())
            print(metrics)


if __name__ == '__main__':
    main()
