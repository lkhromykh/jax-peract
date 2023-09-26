from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import src.types_ as types
import src.encodings as enc
from src.config import Config

Array = types.Array
_dki = nn.initializers.variance_scaling(
    scale=.4,
    mode='fan_in',
    distribution='truncated_normal')
act = nn.gelu


def layer_norm(x: Array) -> Array:
    ln = nn.LayerNorm()
    return ln(x)


def scaled_dot_product(query: Array,
                       key: Array,
                       value: Array,
                       ) -> Array:
    attn = jnp.einsum('...qhd,...khd->...hqk', query, key)
    attn /= np.sqrt(query.shape[-1])
    attn = jax.nn.softmax(attn)
    return jnp.einsum('...hqk,...khd->...qhd', attn, value)


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
        value = scaled_dot_product(query, key, value)
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
        # TODO: properly restore input/output shapes as in the paper.
        output_channels = inputs_q.shape[-1]
        input_channels = min(output_channels, inputs_kv.shape[-1])
        mha = MultiHeadAttention(self.num_heads,
                                 qk_channels=input_channels,
                                 output_channels=output_channels)
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
        return jnp.mean(latent, axis=-2)

    def initial_latent(self, batch_size: int | None = None) -> Array:
        shape = (self.latent_dim, self.latent_channels)
        init = nn.initializers.variance_scaling(
            self.prior_initial_scale,
            mode='fan_in',
            distribution='truncated_normal'
        )
        prior = self.param('prior', init, shape)
        if batch_size is not None:
            prior = jnp.repeat(prior[jnp.newaxis], batch_size, 0)
        return prior


class ObsPreprocessor(nn.Module):

    obs_spec: types.ObservationSpec
    dim: int
    num_bands: int

    @nn.compact
    def __call__(self, obs: types.Observation) -> Array:
        # 1. Preprocess inputs
        # 2. Add positional encodings
        # 3. Add modality encoding
        # 4. concatenate in a single array
        known_modalities = {
            'voxels': self._convs,
            'low_dim': self._mlp,
            'task': self._mlp
        }
        obs = self._maybe_batch(obs)
        fused = []
        for key, fn in known_modalities.items():
            obs[key] = fn(obs[key])
        for key, val in enc.multimodal_encoding(obs).items():
            val = jnp.concatenate([obs[key], val], -1)
            fused.append(val)
        return jnp.concatenate(fused, 1)

    def _convs(self, voxel_grid: Array) -> Array:
        x = voxel_grid / 255.
        # for _ in range(3):
            # x = nn.Conv(self.dim, (3, 3, 3), 2,
            #             use_bias=False, padding='VALID')(x)
            # x = layer_norm(x)
            # x = act(x)
        x = self._positional_encoding(x)
        x = nn.Conv(self.dim, (1, 1, 1))(x)
        return jnp.reshape(x, (x.shape[0], -1, self.dim))

    def _mlp(self, x: Array) -> Array:
        x = jnp.expand_dims(x, -1)
        x = self._positional_encoding(x)
        return nn.Dense(self.dim)(x)

    def _positional_encoding(self, x: Array) -> Array:
        # lower batched input assumption
        batch, *nyquist_freqs = x.shape[:-1]
        axis = range(1, 1 + len(nyquist_freqs))
        pos_enc = enc.positional_encoding(
            x, axis, self.num_bands, nyquist_freqs)
        pos_enc = jnp.repeat(pos_enc[jnp.newaxis], batch, 0)
        return jnp.concatenate([x, pos_enc], -1)

    def _maybe_batch(self, obs: types.Observation) -> types.Observation:
        if obs['low_dim'].ndim == 1:
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis], obs)
        return obs


class ActPreprocess(nn.Module):

    act_spec: types.ActionSpec

    @nn.compact
    def __call__(self, state: Array) -> tfd.Distribution:
        nbins = list(map(lambda sp: sp.num_values, self.act_spec))
        logits = nn.Dense(sum(nbins))(state)
        *logits, _ = jnp.split(logits, np.cumsum(nbins), -1)
        return tfd.Blockwise([tfd.Categorical(log) for log in logits])


class Networks(nn.Module):

    config: Config
    obs_spec: types.ObservationSpec
    act_spec: types.ActionSpec

    def setup(self) -> None:
        c = self.config
        self.preprocessor = ObsPreprocessor(
            self.obs_spec,
            c.latent_dim,
            c.num_bands
        )
        self.encoder = Perceiver(
            c.latent_dim,
            c.latent_channels,
            c.cross_attend_blocks,
            c.self_attend_blocks,
            c.cross_attend_heads,
            c.self_attend_heads,
            c.cross_attend_widening_factor,
            c.self_attend_widening_factor,
            c.prior_initial_scale
        )
        self.postprocessor = ActPreprocess(self.act_spec)

    @nn.compact
    def __call__(self, obs: types.Observation) -> tfd.Distribution:
        x = self.preprocessor(obs)
        x = self.encoder(x)
        return self.postprocessor(x)
