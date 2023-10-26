import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from src import ops
import src.types_ as types
from src.config import Config
from src import networks as nets


class PerAct(nn.Module):

    cfg: Config
    observation_spec: types.ObservationSpec
    action_spec: types.ActionSpec

    def setup(self) -> None:
        c = self.cfg
        self.vgrid_proc = nets.VoxelGridProcessor(
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
            [obs['voxels'], obs['low_dim'], obs['task']],
            [4, 1, 1])
        chex.assert_type(
            [obs['voxels'], obs['low_dim'], obs['task']],
            [jnp.int32, float, jnp.int32]
        )
        c = self.cfg

        vgrid, skip_connections = self.vgrid_proc.encode(obs['voxels'])
        pos3d_enc = ops.fourier_features(vgrid, range(3), c.ff_num_bands)
        vgrid = jnp.concatenate([vgrid, pos3d_enc], -1)
        vgrid = vgrid.reshape(-1, vgrid.shape[-1])
        low_dim = obs['low_dim'].reshape(1, -1)
        task = obs['task'].reshape(1, -1)

        inputs_q = nets.InputsMultiplexer(c.prior_initial_scale)(
            vgrid, low_dim, task
        )
        outputs_q = nets.InputsMultiplexer(c.prior_initial_scale)(
            vgrid, low_dim
        )
        outputs_q = self.perceiver(inputs_q, outputs_q)
        vgrid = outputs_q[:-1].reshape(vgrid.shape[:-1])
        vgrid = self.vgrid_proc.decode(vgrid, skip_connections)
        dist = self.action_decoder(vgrid, outputs_q[-1])
        return dist

