import time

import jax
import chex
chex.disable_asserts()

from src.config import Config
from src.builder import Builder
from src.logger import get_logger
from src.dataset.dataset import DemosDataset
from rltools.loggers import TFSummaryLogger


def _debug():
    import flax
    import logging
    get_logger().setLevel(logging.DEBUG)
    jax.config.update('jax_disable_jit', True)
    # jax.config.update('jax_platform_name', 'cpu')
    chex.enable_asserts()
    flax.linen.enable_named_call()


# TODO: subproc this
def collect_dataset(cfg: Config):
    from src.environment.rlbench_env import RLBenchEnv
    builder = Builder(cfg)
    enc = builder.make_encoders()
    env = builder.make_env(enc).env
    ds = DemosDataset(cfg.dataset_dir)
    logger = get_logger()
    assert isinstance(env, RLBenchEnv)
    tasks = RLBenchEnv.TASKS
    for task in tasks:
        env.TASKS = (task,)
        for _ in range(cfg.num_demos_per_task):
            success = False
            while not success:
                try:
                    desc = env.reset().observation.goal.item()
                    logger.info('Task: %s', desc)
                    demo = env.get_demo()
                    success = True
                except Exception as exc:
                    logger.error('Demo exception: %s', exc)
            ds.append(demo)
            logger.info('ep_length %d, total_eps %d', len(demo), len(ds))
    env.close()


# TODO: profile execution
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
    logger = get_logger()

    def act(obs):
        policy = jax.jit(nets.apply)(params, obs)
        return jax.device_get(policy.mode())

    def env_loop():
        ts = env.reset()
        logger.info('Goal: %s', env.env.get_goal())
        reward = 0
        while not ts.last():
            action = act(ts.observation)
            logger.info('Action %s / %s', enc.action_encoder.decode(action), action)
            ts = env.step(action)
            reward += ts.reward
        logger.info('Reward: %f', reward)
        return reward
    res = [env_loop() for _ in range(100)]
    logger.info(res)
    logger.info('Mean reward: %.3f', float(sum(res)) / len(res))
    env.close()


if __name__ == '__main__':
    # _debug()
    _cfg = Config()
    collect_dataset(_cfg)
    # train(_cfg)
    # evaluate(_cfg)
