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
_dki = nn.initializers.lecun_normal()
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

    layers: types.Layers
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer, kernel_init=_dki)(x)
            if i != len(self.layers) - 1 or self.activate_final:
                x = act(x)
        return x


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
    feedforward_dim: int

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
        mlp = MLP((self.feedforward_dim, output_channels))
        return x + mlp(layer_norm(x))


class SelfAttention(nn.Module):

    num_heads: int
    feedforward_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        xln = layer_norm(x)
        mha = MultiHeadAttention(self.num_heads)
        val = mha(xln, xln)
        x += val
        mlp = MLP((self.feedforward_dim, x.shape[-1]))
        return x + mlp(layer_norm(x))


class Perceiver(nn.Module):

    num_heads: int
    num_blocks: int
    feedforward_dim: int
    latent_dim: int
    latent_channels: int
    lt_num_heads: int
    lt_feedforward_dim: int
    lt_num_blocks: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        def ca_factory():
            return CrossAttention(self.num_heads, self.feedforward_dim)

        def sa_factory():
            return SelfAttention(self.lt_num_heads, self.lt_feedforward_dim)

        latent = self.initial_latent(x.shape[0])
        latent = ca_factory()(latent, x)
        cross_attention = ca_factory()
        latent_transformer = nn.Sequential(
            [sa_factory() for _ in range(self.lt_num_blocks)]
        )
        latent = act(latent)
        for i in range(self.num_blocks):
            if i: latent = cross_attention(latent, x)
            latent = latent_transformer(latent)
        latent = act(latent)
        return jnp.mean(latent, axis=-2)

    def initial_latent(self, batch_size: int | None = None) -> Array:
        shape = (self.latent_channels, self.latent_dim)
        prior = self.param('prior', nn.initializers.lecun_normal(), shape)
        if batch_size is not None:
            prior = jnp.repeat(prior[jnp.newaxis], batch_size, 0)
        return prior


class ObsPreprocessor(nn.Module):

    dim: int
    num_freq_bands: int
    conv_filters: types.Layers

    @nn.compact
    def __call__(self, obs: types.Observation) -> Array:
        # 1. Preprocess inputs
        # 2. Add positional encodings
        # 3. Add modality encoding
        # 4. concatenate in a single array
        known_modalities = {
            'voxels': self._convs,
            'low_dim': self._mlp
        }
        fused = []
        obs = self._maybe_batch(obs)
        for key, val in obs.items():
            obs[key] = known_modalities[key](val)
        for key, val in enc.multimodal_encoding(obs).items():
            val = jnp.concatenate([obs[key], val], -1)
            fused.append(val)
        return jnp.concatenate(fused, 1)

    def _convs(self, voxel_grid: Array) -> Array:
        x = voxel_grid / 255.
        for dim in self.conv_filters:
            x = nn.Conv(dim, (3, 3, 3), 2, padding='VALID')(x)
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
        batch, *nyquist_freqs = x.shape[:-1]
        axis = range(1, 1 + len(nyquist_freqs))
        pos_enc = enc.positional_encoding(
            x, axis, self.num_freq_bands, nyquist_freqs)
        pos_enc = jnp.repeat(pos_enc[jnp.newaxis], batch, 0)
        return jnp.concatenate([x, pos_enc], -1)

    def _maybe_batch(self, obs: types.Observation) -> types.Observation:
        if obs['low_dim'].ndim == 1:
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis], obs)
        return obs


class Networks(nn.Module):

    config: Config

    def setup(self) -> None:
        c = self.config
        self.preprocessor = ObsPreprocessor(31, 5, (33,))
        self.encoder = Perceiver(
            c.num_heads,
            c.num_blocks,
            c.feedforward_dim,
            c.latent_dim,
            c.latent_channels,
            c.lt_num_heads,
            c.lt_feedforward_dim,
            c.lt_num_blocks,
        )

    @nn.compact
    def __call__(self, obs: types.Observation) -> tfd.Distribution:
        x = self.preprocessor(obs)
        x = self.encoder(x)
        logits = nn.Dense(8 * self.config.nbins)(x)
        logits = logits.reshape(logits.shape[:-1] + (8, self.config.nbins))
        dist = tfd.Categorical(logits)
        return tfd.Independent(dist, 1)
