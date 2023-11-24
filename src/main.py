import logging
import warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

import jax
jax.config.update('jax_platform_name', 'cpu')

from src.config import Config
from src.builder import Builder
from src.rlbench_env.enviroment import environment_loop
from rltools.loggers import TFSummaryLogger


def main():
    cfg = Config()
    builder = Builder(cfg)
    rngseq = iter(jax.random.split(jax.random.PRNGKey(cfg.seed), 4))
    env = builder.make_env(next(rngseq)) if cfg.launch_env else None
    ds, specs = builder.make_dataset_and_specs(next(rngseq), env)
    nets, params = builder.make_networks_and_params(next(rngseq), specs)
    step = builder.make_step_fn(nets)
    apply = jax.jit(nets.apply)
    state = builder.make_state(next(rngseq), params)
    state = jax.device_put(state)

    logger = TFSummaryLogger(logdir=cfg.logdir, label='bc', step_key='step')

    for t in range(cfg.training_steps):
        batch = jax.device_put(next(ds))
        state, metrics = step(state, batch)
        if t % cfg.eval_every == 0:
            metrics.update(step=t)
            if cfg.launch_env:
                def policy(obs):
                    action = apply(state.params, obs).mode()
                    return jax.device_get(action)
                reward = environment_loop(policy, env)
                metrics.update(eval_reward=reward)
            logger.write(metrics)
            builder.save(jax.device_get(state), Builder.STATE)


if __name__ == '__main__':
    main()
