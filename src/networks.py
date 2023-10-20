from typing import Callable

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

Array = jax.Array
_dki = nn.initializers.variance_scaling(
    scale=.4,
    mode='fan_in',
    distribution='truncated_normal')
act = nn.gelu


def layer_norm(x: Array) -> Array:
    ln = nn.LayerNorm()
    return ln(x)


class MLP(nn.Module):

    widening_factor: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        dim = x.shape[-1]
        x = nn.Dense(self.widening_factor * dim, kernel_init=_dki)(x)
        x = act(x)
        return nn.Dense(dim, kernel_init=_dki)(x)


class MultiHeadAttention(nn.Module):

    num_heads: int
    qk_channels: int | None = None
    v_channels: int | None = None
    output_channels: int | None = None

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 ) -> Array:
        def dense(dim, name):
            dim, res = np.divmod(dim, self.num_heads)
            assert res == 0, f'Not divisible by the number of heads: {dim}.'
            return nn.DenseGeneral(features=(self.num_heads, dim),
                                   name=name,
                                   kernel_init=_dki
                                   )
        qk_channels = self.qk_channels or inputs_q.shape[-1]
        v_channels = self.v_channels or qk_channels
        output_channels = self.output_channels or v_channels

        query = dense(qk_channels, name='query')(inputs_q)
        key = dense(qk_channels, name='key')(inputs_kv)
        value = dense(v_channels, name='value')(inputs_kv)
        value = ops.scaled_dot_product(query, key, value)
        proj = nn.DenseGeneral(output_channels,
                               axis=(-2, -1),
                               kernel_init=_dki,
                               name='proj')
        return proj(value)


class CrossAttention(nn.Module):

    num_heads: int
    widening_factor: int

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 ) -> Array:
        channels = inputs_q.shape[-1]
        mha = MultiHeadAttention(self.num_heads,
                                 qk_channels=channels,
                                 output_channels=channels)
        val = mha(layer_norm(inputs_q), layer_norm(inputs_kv))
        x = inputs_q + val
        mlp = MLP(self.widening_factor)
        return x + mlp(layer_norm(x))


class SelfAttention(nn.Module):

    num_heads: int
    widening_factor: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        xln = layer_norm(x)
        mha = MultiHeadAttention(self.num_heads)
        val = mha(xln, xln)
        x += val
        mlp = MLP(self.widening_factor)
        return x + mlp(layer_norm(x))


class Perceiver(nn.Module):

    latent_dim: int
    latent_channels: int
    cross_attend_blocks: int
    self_attend_blocks: int
    cross_attend_heads: int
    self_attend_heads: int
    cross_attend_widening_factor: int
    self_attend_widening_factor: int
    prior_initial_scale: float = 0.02

    @nn.compact
    def __call__(self, x: Array) -> Array:
        chex.assert_type(x, float)
        chex.assert_rank(x, 3)  # (batch, sequence_len, channel_dim)

        def ca_factory():
            return CrossAttention(self.cross_attend_heads,
                                  self.cross_attend_widening_factor)

        def sa_factory():
            return SelfAttention(self.self_attend_heads,
                                 self.self_attend_widening_factor)

        latent = self.initial_latent(x.shape[0])
        latent = ca_factory()(latent, x)
        cross_attention = ca_factory()
        latent_transformer = nn.Sequential(
            [sa_factory() for _ in range(self.self_attend_blocks)]
        )
        for i in range(self.cross_attend_blocks):
            if i: latent = cross_attention(latent, x)
            latent = latent_transformer(latent)
        return jnp.reshape(latent, latent.shape[:-2] + (-1,))

    def initial_latent(self, batch_size: int) -> Array:
        shape = (self.latent_dim, self.latent_channels)
        init = nn.initializers.variance_scaling(
            self.prior_initial_scale,
            mode='fan_in',
            distribution='truncated_normal'
        )
        prior = self.param('prior', init, shape)
        prior = jnp.repeat(prior[jnp.newaxis], batch_size, 0)
        return prior


class VoxelGridProcessor(nn.Module):

    features: types.Layers
    kernels: types.Layers
    strides: types.Layers

    def setup(self) -> None:
        self.convs = self._make_stem(nn.Conv)
        self.deconvs = self._make_stem(nn.ConvTranspose)

    def encode(self, voxel_grid: Array) -> tuple[Array, list[Array]]:
        chex.assert_type(voxel_grid, int)
        chex.assert_rank(voxel_grid, 4)

        x = voxel_grid / 128. - 1
        skip_connections = []
        for block in self.convs:
            x = block(x)
            skip_connections.append(x)
        return x, skip_connections

    def decode(self, x: Array, skip_connections: list[Array]) -> Array:
        chex.assert_type(x, float)
        chex.assert_rank(x, 4)

        for block, y in reversed(zip(self.deconvs, skip_connections)):
            x = jnp.concatenate([x, y], -1)
            x = block(x)
        return x

    def _make_stem(self, conv_cls: nn.Conv | nn.ConvTranspose) -> nn.Sequential:
        blocks = []
        arch = zip(self.features, self.kernels, self.strides)
        for f, k, s in arch:
            conv = conv_cls(features=f,
                            kernel_size=3 * (k,),
                            strides=3 * (s,),
                            use_bias=False,
                            padding='VALID'
                            )
            block = nn.Sequential([conv,
                                   nn.LayerNorm(),
                                   act
                                   ])
            blocks.append(block)
        return blocks


class ObsMultiplexer(nn.Module):

    @nn.compact
    def __call__(self, obs: types.Observation) -> Array:
        chex.assert_rank(obs.values(), 3)  # B x {L} x {C}
        maxlen = max(map(lambda x: x.shape[-1], obs.values()))
        maxlen += 4  # force minimal padding
        for k, v in sorted(obs.items()):
            obs[k] = nn.Dense(maxlen)(v)
        return jnp.concatenate(list(obs.values()), 1)


class ActProcessor(nn.Module):

    act_spec: types.ActionSpec
    vgrid_decoder: VoxelGridProcessor

    class Blockwise(tfd.Blockwise):
        def mode(self, *args, **kwargs):
            mode = map(lambda x: x.mode(*args, **kwargs), self.distributions)
            return jnp.stack(list(mode), -1)

    @nn.compact
    def __call__(self,
                 latent: Array,
                 skip_connections: tuple[Array]
                 ) -> tfd.Distribution:
        nbins = map(lambda x: x.num_values, self.act_spec)
        logits = nn.Dense(sum(nbins))(latent)
        vgrid, *logits, _ = jnp.split(logits, np.cumsum(nbins), -1)
        vgrid = self._decode_voxels_grid(vgrid, skip_connections)
        logits = (vgrid,) + logits
        return ActProcessor.Blockwise([tfd.Categorical(log) for log in logits])

    def _decode_voxels_grid(self, vgrid: Array,
                            skip_connections: tuple[Array]
                            ) -> Array:
        scene_bins = np.cbrt(self.act_spec[0].num_values).astype(int)
        vgrid = jnp.reshape(vgrid, (-1,) + 3 * (scene_bins,))
        vgrid = self.vgrid_decoder(vgrid, skip_connections)
        assert np.prod(vgrid.shape) == scene_bins
        return vgrid


class Networks(nn.Module):

    config: Config
    obs_spec: types.ObservationSpec
    act_spec: types.ActionSpec

    def setup(self) -> None:
        c = self.config
        # Subtract #modalities to actually match latent_channels dim.
        num_modalities = len(self.obs_spec)
        # self.preprocessor = ObsPreprocessor(
        #     self.obs_spec,
        #     c.latent_channels - num_modalities,
        #     c.num_bands
        # )
        # self.encoder = Perceiver(
        #     c.latent_dim,
        #     c.latent_channels,
        #     c.cross_attend_blocks,
        #     c.self_attend_blocks,
        #     c.cross_attend_heads,
        #     c.self_attend_heads,
        #     c.cross_attend_widening_factor,
        #     c.self_attend_widening_factor,
        #     c.prior_initial_scale
        # )
        self.postprocessor = ActProcessor(self.act_spec)

    @nn.compact
    def __call__(self, obs: types.Observation) -> tfd.Distribution:
        x = MLP(64)(jnp.atleast_2d(obs['low_dim']))
        # x = self.preprocessor(obs)
        # x = self.encoder(x)
        return self.postprocessor(x)


class PerAct(nn.Module):

    cfg: Config
    observation_spec: types.ObservationSpec
    action_spec: types.ActionSpec

    def setup(self) -> None:
        self.vgrid_proc = VoxelGridProcessor()

    @nn.compact
    def __call__(self, obs: types.Observation) -> tfd.Distribution:
        ...
