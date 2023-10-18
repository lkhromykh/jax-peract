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

    features: int
    num_blocks: int

    @nn.compact
    def __call__(self, *args):
        # TODO: refactor this: method should be supplied instead.
        if self.is_initializing():
            return self.decode(*self.encode(*args))
        match len(args):
            case 1: return self.encode(*args)
            case 2: return self.decode(*args)
            case _: raise ValueError

    def encode(self, voxel_grid: Array) -> tuple[Array, list[Array]]:
        chex.assert_type(voxel_grid, int)
        chex.assert_rank(voxel_grid, 4)

        x = (voxel_grid - 128) / 255.
        skip_connections = []
        for _ in range(self.num_blocks):
            x = self._conv_block(nn.Conv)(x)
            skip_connections.append(x)
        x = jnp.reshape(x, (-1,))
        return x, skip_connections

    def decode(self, latent: Array, skip_connections: list[Array]) -> Array:
        chex.assert_type(latent, float)
        chex.assert_rank(latent, 2)

        shape = skip_connections[-1].shape
        x = nn.Dense(np.prod(shape))(latent)
        x = jnp.reshape(x, shape)
        for y in reversed(skip_connections):
            x = jnp.concatenate([x, y], -1)
            x = self._conv_block(nn.ConvTranspose)(x)
        return x

    def _conv_block(self, conv: nn.Conv | nn.ConvTranspose) -> nn.Sequential:
        return nn.Sequential([
            conv(features=self.features,
                 kernel_size=(3, 3, 3),
                 strides=(2, 2, 2),
                 use_bias=False,
                 padding='VALID'),
            nn.LayerNorm(),
            nn.relu
        ])


class ObsProcessor(nn.Module):

    modality_processors: dict[str, Callable]
    dim: int
    num_bands: int

    @nn.compact
    def __call__(self, obs: types.Observation) -> Array:
        # 1. Preprocess inputs
        # 2. Add positional encodings
        # 3. Add learned modality specific embedding.
        # 4. concatenate in a single array
        obs = self._maybe_batch(obs)
        max_dim = -1
        out = {}
        breakpoint()
        for key, fn in self.modality_processors.items():
            val = fn(obs[key])
            out[key] = val
        for key, val in ops.multimodal_encoding(out).items():
            out[key] = jnp.concatenate([out[key], val], -1)
        return jnp.concatenate(list(out.values()), 1)

    def _convs(self, voxel_grid: Array) -> Array:
        x = voxel_grid / 128. - 1
        for _ in range(2):
            x = nn.Conv(self.dim, (3, 3, 3), 2,
                        use_bias=False, padding='VALID')(x)
            x = layer_norm(x)
            x = act(x)
        x = self._positional_encoding(x)
        x = nn.Conv(self.dim, (1, 1, 1))(x)
        return jnp.reshape(x, (x.shape[0], -1, self.dim))

    def _mlp(self, x: Array) -> Array:
        x = jnp.expand_dims(x, -1)
        x = self._positional_encoding(x)
        return nn.Dense(self.dim)(x)

    def _positional_encoding(self, x: Array) -> Array:
        # lower batched input assumption
        breakpoint()
        batch, *nyquist_freqs = x.shape[:-1]
        axes = range(1, 1 + len(nyquist_freqs))
        pos_enc = ops.positional_encoding(
            x, axes, self.num_bands, nyquist_freqs)
        pos_enc = np.repeat(pos_enc[np.newaxis], batch, 0)
        return jnp.concatenate([x, pos_enc], -1)

    def _maybe_batch(self, obs: types.Observation) -> types.Observation:
        if obs['low_dim'].ndim == 1:
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis], obs)
        return obs


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
        vgrid = self.vgrid_decoder()
        logits = nn.Dense(sum(nbins))(state)
        *logits, _ = jnp.split(logits, np.cumsum(nbins), -1)
        return ActProcessor.Blockwise([tfd.Categorical(log) for log in logits])


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
