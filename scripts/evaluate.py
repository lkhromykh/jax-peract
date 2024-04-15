import os
import sys
import pathlib
import multiprocessing as mp
from functools import partial
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import jax

from peract.config import Config
from peract.builder import Builder
from peract.logger import get_logger
from peract.environment import RLBenchEnv


def evaluate_one(cfg: Config, task: str | None = None):
    builder = Builder(cfg)
    enc = builder.make_encoders()
    nets, _ = builder.make_networks_and_params(enc)
    params = builder.load(Builder.STATE).params
    env = builder.make_env(enc)
    if task is not None:
        assert isinstance(env, RLBenchEnv)
        env.TASKS = (task,)
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
    logger.info('Task %s mean reward: %.3f', task, float(sum(res)) / len(res))
    env.close()


def mp_evaluate(cfg: Config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    evaluate = partial(evaluate_one, cfg)
    tasks = RLBenchEnv.TASKS
    with mp.Pool(len(tasks)) as p:
        p.map(evaluate, tasks)


def sync_evaluate(cfg: Config):
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    for task in RLBenchEnv.TASKS:
        jax.clear_caches()
        jax.clear_backends()
        evaluate_one(cfg, task)


if __name__ == '__main__':
    exp_path = pathlib.Path(sys.argv[1])
    cfg_ = Config.load(exp_path / Builder.CONFIG, compute_dtype='f32')
    evaluate_one(cfg_)
