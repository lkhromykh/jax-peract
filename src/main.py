import jax
jax.config.update('jax_platform_name', 'cpu')

from src.config import Config
from src.builder import Builder
from src.rlbench_env.dataset import as_tfdataset
from src.rlbench_env.enviroment import environment_loop


def main():
    cfg = Config()
    builder = Builder(cfg)
    env = builder.make_env(cfg.seed)
    ts = env.reset()
    optim = builder.make_optim()
    nets = builder.make_networks(env)
    step = builder.make_step_fn(nets)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, subkey = jax.random.split(rng, 2)
    params = nets.init(subkey, ts.observation)
    state = builder.make_state(rng, params)

    ds = as_tfdataset(env.get_demos(5))\
        .repeat()\
        .batch(cfg.batch_size)\
        .prefetch(-1)
    ds = ds.as_numpy_iterator()

    for batch in ds:
        batch = jax.device_put(batch)
        state, metrics = step(state, batch)
        t = state.step.item()
        if t % 100 == 0:
            policy = lambda obs: nets.apply(state.params, obs).sample(seed=rng)
            reward = environment_loop(policy, env)
            metrics.update(step=t, eval_reward=reward)
            print(metrics)


if __name__ == '__main__':
    main()
