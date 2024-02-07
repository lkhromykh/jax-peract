import time
import logging
logging.basicConfig(level=logging.INFO)

import jax
import chex
chex.disable_asserts()

from src.config import Config
from src.builder import Builder
from src.dataset.dataset import DemosDataset
from rltools.loggers import TFSummaryLogger


def _debug():
    import flax
    jax.config.update('jax_disable_jit', True)
    # jax.config.update('jax_platform_name', 'cpu')
    chex.enable_asserts()
    flax.linen.enable_named_call()


def collect_dataset(cfg: Config):
    builder = Builder(cfg)
    env = builder.make_env().env
    ds = DemosDataset(cfg.dataset_dir)
    while len(ds) < cfg.num_demos:
        logging.info(f'Demo {len(ds)}')
        env.reset()
        demo = env.get_demo()
        ds.append(demo)
    env.close()


def train(cfg: Config):
    builder = Builder(cfg)
    ds = builder.make_tfdataset().as_numpy_iterator()
    nets, params = builder.make_networks_and_params()
    step = builder.make_step_fn(nets)
    state = builder.make_state(params)
    state = jax.device_put(state)
    logger = TFSummaryLogger(logdir=cfg.logdir, label='bc', step_key='step')

    start = time.time()
    t = state.step.item()
    while t < cfg.training_steps:
        batch = jax.device_put(next(ds))
        state, metrics = step(state, batch)
        if t % cfg.log_every == 0:
            fps = float(cfg.batch_size) * t / (time.time() - start)
            metrics.update(step=t, fps=fps)
            logger.write(metrics)
        if t % cfg.save_every == 0:
            builder.save(jax.device_get(state), Builder.STATE)
        t = state.step.item()


def evaluate(cfg: Config):
    builder = Builder(cfg)
    nets, _ = builder.make_networks_and_params()
    params = builder.load(Builder.STATE).params
    env = builder.make_env()
    def act(obs): return jax.jit(nets.apply)(params, obs).mode()

    def env_loop():
        ts = env.reset()
        reward = 0
        while not ts.last():
            action = act(ts.observation)
            ts = env.step(action)
            reward += ts.reward
        return reward
    res = [env_loop() for _ in range(10)]
    env.close()
    logging.info(str(res))
    print(res)


if __name__ == '__main__':
    # _debug()
    _cfg = Config()
    collect_dataset(_cfg)
    train(_cfg)
    evaluate(_cfg)
