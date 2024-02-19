import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import jax
import chex
import psutil
import GPUtil
chex.disable_asserts()

from src.config import Config
from src.builder import Builder
from src.logger import get_logger
from rltools.loggers import TFSummaryLogger


def _debug():
    import flax
    import logging
    get_logger().setLevel(logging.DEBUG)
    jax.config.update('jax_disable_jit', True)
    # jax.config.update('jax_platform_name', 'cpu')
    chex.enable_asserts()
    flax.linen.enable_named_call()


def train(cfg: Config):
    builder = Builder(cfg)
    enc = builder.make_encoders()
    ds = builder.make_tfdataset().as_numpy_iterator()
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
            metrics.update(step=t,
                           fps=fps,
                           util_cpu_percent=psutil.cpu_percent(),
                           util_mem_precent=psutil.virtual_memory().percent,
                           util_gpu_load=GPUtil.getGPUs()[0].load,
                           )
            logger.write(metrics)
        if t % cfg.save_every == 0:
            builder.save(jax.device_get(state), Builder.STATE)


if __name__ == '__main__':
    # _debug()
    _cfg = Config()
    train(_cfg)
