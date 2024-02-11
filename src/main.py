import time
import logging
logging.basicConfig(level=logging.DEBUG)

import jax
import chex
chex.disable_asserts()

from src.config import Config
from src.builder import Builder
from src.dataset.dataset import DemosDataset
from rltools.loggers import TFSummaryLogger


def _debug():
    import flax
    logging.basicConfig(level=logging.DEBUG)
    jax.config.update('jax_disable_jit', True)
    # jax.config.update('jax_platform_name', 'cpu')
    chex.enable_asserts()
    flax.linen.enable_named_call()


def collect_dataset(cfg: Config):
    builder = Builder(cfg)
    enc = builder.make_encoders()
    env = builder.make_env(enc).env
    ds = DemosDataset(cfg.dataset_dir)
    while len(ds) < cfg.num_demos:
        env.reset()
        demo = env.get_demo()
        ds.append(demo)
        logging.info('Demo %d length %d', len(ds), len(demo))
    env.close()


def train(cfg: Config):
    builder = Builder(cfg)
    enc = builder.make_encoders()
    ds = builder.make_tfdataset(enc).as_numpy_iterator()
    nets, params = builder.make_networks_and_params(enc)
    step = builder.make_step_fn(nets)
    state = builder.make_state(params)
    state = jax.device_put(state)
    logger = TFSummaryLogger(logdir=cfg.logdir, label='bc', step_key='step')

    t = state.step.item()
    while t < cfg.training_steps:
        _batch_start = time.time()
        batch = jax.device_put(next(ds))
        state, metrics = step(state, batch)
        t = state.step.item()
        if t % cfg.log_every == 0:
            state, metrics = jax.block_until_ready((state, metrics))
            fps = float(cfg.batch_size) / (time.time() - _batch_start)
            metrics.update(step=t, fps=fps)
            logger.write(metrics)
        if t % cfg.save_every == 0:
            builder.save(jax.device_get(state), Builder.STATE)


def evaluate(cfg: Config):
    builder = Builder(cfg)
    enc = builder.make_encoders()
    nets, _ = builder.make_networks_and_params(enc)
    params = builder.load(Builder.STATE).params
    env = builder.make_env(enc)

    def act(obs):
        policy = jax.jit(nets.apply)(params, obs)
        return jax.device_get(policy.mode())

    def env_loop():
        ts = env.reset()
        logging.debug('Goal: %s', env.env.get_goal())
        reward = 0
        while not ts.last():
            action = act(ts.observation)
            logging.debug('Action %s / %s', enc.action_encoder.decode(action), action)
            ts = env.step(action)
            reward += ts.reward
        logging.debug('Reward: %f', reward)
        return reward
    res = [env_loop() for _ in range(50)]
    logging.debug(res)
    logging.debug('Mean reward: %.3f', float(sum(res)) / len(res))
    env.close()


if __name__ == '__main__':
    # _debug()
    _cfg = Config()
    # collect_dataset(_cfg)
    train(_cfg)
    evaluate(_cfg)
