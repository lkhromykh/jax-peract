import logging
import warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

import cloudpickle
import jax
# jax.config.update('jax_platform_name', 'cpu')

from src.config import Config
from src.builder import Builder
from src.rlbench_env.enviroment import environment_loop
from rltools.loggers import TFSummaryLogger


def main():
    cfg = Config()
    builder = Builder(cfg)
    rngs = jax.random.split(jax.random.PRNGKey(cfg.seed), 3)
    env = builder.make_env(rngs[0])
    ds = builder.make_dataset(env)
    nets, params = builder.make_networks_and_params(rngs[1], env)
    step = builder.make_step_fn(nets)
    apply = jax.jit(nets.apply)
    state = builder.make_state(rngs[2], params)
    state = jax.device_put(state)

    logger = TFSummaryLogger(logdir=cfg.logdir, label='bc', step_key='step')

    for batch in ds:
        batch = jax.device_put(batch)
        state, metrics = step(state, batch)
        t = state.step.item()
        if t % cfg.eval_every == 0:
            def policy(obs):
                action = apply(state.params, obs).mode()
                return jax.device_get(action)
            reward = environment_loop(policy, env)
            metrics.update(step=t, eval_reward=reward)
            logger.write(metrics)
            with open(builder.exp_path(Builder.STATE), 'wb') as f:
                cloudpickle.dump(jax.device_get(state), f)


if __name__ == '__main__':
    main()
