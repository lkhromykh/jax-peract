import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import jax
import chex
import psutil
import GPUtil
from rltools.loggers import TFSummaryLogger
chex.disable_asserts()

from src.config import Config
from src.builder import Builder
from src.logger import get_logger
from src.utils import prefetch_to_device


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
    train_ds = builder.make_tfdataset('train')
    train_ds = prefetch_to_device(train_ds.as_numpy_iterator())
    val_ds = builder.make_tfdataset('val')
    nets, params = builder.make_networks_and_params(enc)
    train_step = builder.make_step_fn(nets, 'train')
    val_step = builder.make_step_fn(nets, 'val')
    state = builder.make_state(params)
    state = jax.device_put(state)
    train_logger = TFSummaryLogger(logdir=cfg.logdir, label='train', step_key='step')
    val_logger = TFSummaryLogger(logdir=cfg.logdir, label='val', step_key='step')

    t = state.step.item()
    while t < cfg.training_steps:
        _batch_start = time.time()
        state, metrics = train_step(state, next(train_ds))
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
            train_logger.write(metrics)
        if t % cfg.save_every == 0:
            batch_metrics = []
            for batch in prefetch_to_device(val_ds.as_numpy_iterator()):
                batch = jax.device_put(batch)
                _, metrics = val_step(state, batch)
                batch_metrics.append(metrics)
            metrics = jax.tree_util.tree_map(lambda *ts: jax.numpy.stack(ts), *batch_metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.mean(0), metrics)
            metrics.update(step=t)
            val_logger.write(metrics)
            builder.save(jax.device_get(state), Builder.STATE)


if __name__ == '__main__':
    # _debug()
    train(Config())
