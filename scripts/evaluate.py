import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import jax

from src.config import Config
from src.builder import Builder
from src.logger import get_logger


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
    evaluate(Config())
