import warnings
warnings.filterwarnings('ignore')

import cloudpickle
import jax
# jax.config.update('jax_platform_name', 'cpu')

from src.config import Config
from src.builder import Builder
from src.rlbench_env.dataset import as_tfdataset
from src.rlbench_env.enviroment import environment_loop
from rltools.loggers import TFSummaryLogger

def main():
    cfg = Config()
    builder = Builder(cfg)
    # with open(builder.exp_path(Builder.CONFIG), 'wb') as f:
    #     cloudpickle.dump(cfg, f)
    rngs = jax.random.split(jax.random.PRNGKey(cfg.seed), 3)
    env = builder.make_env(rngs[0])
    nets, params = builder.make_networks_and_params(rngs[1], env)
    step = builder.make_step_fn(nets)
    state = builder.make_state(rngs[2], params)

    ds = as_tfdataset(env.get_demos(20))
    ds = ds.repeat()\
        .batch(cfg.batch_size)\
        .prefetch(-1)
    ds = ds.as_numpy_iterator()
    logger = TFSummaryLogger(logdir=cfg.logdir, label='bc', step_key='step')

    for batch in ds:
        batch = jax.device_put(batch)
        state, metrics = step(state, batch)
        t = state.step.item()
        if t % cfg.eval_every == 0:
            policy = lambda obs: jax.jit(nets.apply)(state.params, obs).mode().squeeze(0)
            reward = environment_loop(policy, env)
            metrics.update(step=t, eval_reward=reward)
            logger.write(metrics)

            # with open(builder.exp_path(Builder.STATE), 'wb') as f:
            #     cloudpickle.dump(state, f)


if __name__ == '__main__':
    main()
