import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import chex
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from src import utils
import src.types_ as types
from src.config import Config
from src import networks as nets


class PerAct(nn.Module):

    cfg: Config
    observation_spec: types.ObservationSpec
    action_spec: types.ActionSpec

    def setup(self) -> None:
        c = self.cfg
        self.voxels_proc = nets.VoxelsProcessor(
            features=c.conv_stem_features,
            kernels=c.conv_stem_kernels,
            strides=c.conv_stem_strides
        )
        self.perceiver = nets.PerceiverIO(
            c.latent_dim,
            c.latent_channels,
            c.num_blocks,
            c.num_self_attend_per_block,
            c.num_cross_attend_heads,
            c.num_self_attend_heads,
            c.cross_attend_widening_factor,
            c.self_attend_widening_factor,
            c.use_query_residual,
            c.prior_initial_scale
        )
        self.action_decoder = nets.ActionDecoder(self.action_spec)

    @nn.compact
    def __call__(self, obs: types.Observation) -> tfd.Distribution:
        chex.assert_rank(
            [obs.voxels, obs.low_dim, obs.task],
            [4, 1, 1])
        chex.assert_type(
            [obs.voxels, obs.low_dim, obs.task],
            [jnp.int32, float, jnp.int32]
        )
        c = self.cfg
        voxels, skip_connections = self.voxels_proc.encode(obs.voxels)
        pos3d_enc = utils.fourier_features(voxels.shape[:-1], c.ff_num_bands)
        voxels = jnp.concatenate([voxels, pos3d_enc], -1)
        voxels = voxels.reshape(-1, voxels.shape[-1])
        low_dim = obs.low_dim.reshape(1, -1)
        task = obs.task.reshape(1, -1)

        inputs_q = nets.InputsMultiplexer(c.prior_initial_scale)(
            voxels, low_dim, task
        )
        outputs_q = nets.InputsMultiplexer(c.prior_initial_scale)(
            voxels, low_dim
        )
        outputs_val = self.perceiver(inputs_q, outputs_q)
        shape = skip_connections[-1].shape[:-1]
        voxels = outputs_val[:-1].reshape(shape + (-1,))
        voxels = self.voxels_proc.decode(voxels, skip_connections)
        return self.action_decoder(voxels, outputs_val[-1])


# class PerAct(nn.Module):
#
#     cfg: Config
#     observation_spec: types.ObservationSpec
#     action_spec: types.ActionSpec
#
#     @nn.compact
#     def __call__(self, obs: types.Observation):
#         nbins = tuple(map(lambda sp: sp.num_values, self.action_spec))
#         x = obs.low_dim
#         for layer in (256, 256):
#             x = nn.Dense(layer)(x)
#             x = nn.tanh(x)
#         logits = nn.Dense(sum(nbins))(x)
#         *logits, _ = jnp.split(logits, np.cumsum(nbins), -1)
#         return nets.ActionDecoder.Blockwise(
#             [tfd.Categorical(logit) for logit in logits]
#         )
