import jax
# jax.config.update('jax_platform_name', 'cpu')

from src.config import Config
from src.builder import Builder
from src import ops


def main():
    cfg = Config()
    builder = Builder(cfg)
    nets = builder.make_networks()
    ds = builder.make_dataset()
    step = builder.make_step_fn(nets)
    img, _ = next(ds)
    rng = jax.random.PRNGKey(cfg.seed)
    params = nets.init(rng, img)
    print('Number of params: ', ops.tree_size(params))
    optim = builder.make_optim()
    state = builder.make_state(rng, params, optim)

    for batch in ds:
        batch = jax.device_put(batch)
        state, metrics = step(state, batch)
        print(metrics)


if __name__ == '__main__':
    main()
