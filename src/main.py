import jax
jax.config.update('jax_platform_name', 'cpu')

from src.config import Config
from src.builder import Builder
from src.rlbench_env.dataset import as_tfdataset


def main():
    cfg = Config()
    builder = Builder(cfg)
    env = builder.make_env('ReachTarget')
    optim = builder.make_optim()
    nets = builder.make_networks()
    step = builder.make_step_fn(nets)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, subkey = jax.random.split(rng, 2)
    params = nets.init(subkey, env.reset().observation)
    state = builder.make_state(rng, params, optim)

    ds = as_tfdataset(env.get_demos(2)).repeat()
    ds = ds.batch(3, drop_remainder=True).as_numpy_iterator()

    for t, batch in enumerate(ds):
        batch = jax.device_put(batch)
        state, metrics = step(state, batch)
        metrics.update(step=t)
        print(metrics)


if __name__ == '__main__':
    main()
