import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import jax
import flax
flax.linen.enable_named_call()
from rltools.loggers import TFSummaryLogger

from src.config import Config
from src.builder import Builder


def profile(cfg: Config):
    builder = Builder(cfg)
    enc = builder.make_encoders()
    ds = builder.make_tfdataset('train').as_numpy_iterator()
    nets, params = builder.make_networks_and_params(enc)
    step = builder.make_step_fn(nets, 'train')
    state = builder.make_state(params)
    state = jax.device_put(state)
    TFSummaryLogger(logdir=cfg.logdir, label='profile', step_key='step')

    # 0. init ds and jit
    batch = jax.device_put(next(ds))
    jax.block_until_ready(step(state, batch))
    # 1. profile
    jax.profiler.start_trace(str(builder.exp_path()))
    batch = jax.device_put(next(ds))
    jax.block_until_ready(step(state, batch))
    jax.profiler.stop_trace()


if __name__ == '__main__':
    _cfg = Config()
    profile(_cfg)
