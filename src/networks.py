from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from src.config import Config
from src.positional_encoding import positional_encoding

Layers = Any
Array = Any
_dki = nn.initializers.lecun_normal()


def layer_norm(x: Array) -> Array:
    ln = nn.LayerNorm()
    return ln(x)


def scaled_dot_product(query: Array,
                       key: Array,
                       value: Array,
                       mask: Array | None = None
                       ) -> tuple[Array, Array]:
    # todo: time_major
    attn = jnp.einsum('...qhd,...khd->...hqk', query, key)
    attn /= np.sqrt(query.shape[-1])
    attn = jax.nn.softmax(attn)
    return jnp.einsum('...hqk,...khd->...qhd', attn, value), attn


class MLP(nn.Module):

    layers: Layers
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer, kernel_init=_dki)(x)
            if i != len(self.layers) - 1 or self.activate_final:
                x = jax.nn.gelu(x)
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
                 mask: Array | None = None
                 ) -> tuple[Array, Array]:
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
        value, attn = scaled_dot_product(query, key, value, mask)
        proj = nn.DenseGeneral(output_channels,
                               axis=(-2, -1),
                               kernel_init=_dki,
                               name='proj')
        return proj(value), attn


class CrossAttention(nn.Module):

    num_heads: int
    feedforward_dim: int

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 mask: Array = None
                 ) -> tuple[Array, Array]:
        # TODO: properly restore input/output shapes as in the paper.
        output_channels = inputs_q.shape[-1]
        input_channels = min(output_channels, inputs_kv.shape[-1])
        mha = MultiHeadAttention(self.num_heads,
                                 qk_channels=input_channels,
                                 output_channels=output_channels)
        val, attn = mha(layer_norm(inputs_q), layer_norm(inputs_kv), mask)
        x = inputs_q + val
        mlp = MLP((self.feedforward_dim, output_channels))
        return x + mlp(layer_norm(x)), attn


class SelfAttention(nn.Module):

    num_heads: int
    feedforward_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        xln = layer_norm(x)
        mha = MultiHeadAttention(self.num_heads)
        val, _ = mha(xln, xln)
        x += val
        mlp = MLP((self.feedforward_dim, x.shape[-1]))
        return x + mlp(layer_norm(x))


# move it to perceiver encoder
class Perceiver(nn.Module):

    config: Config

    @nn.compact
    def __call__(self, x: Array) -> Array:
        c = self.config

        def ca_factory():
            return CrossAttention(c.num_heads, c.feedforward_dim)

        def sa_factory():
            return SelfAttention(c.lt_num_heads, c.lt_feedforward_dim)

        x = self._positional_encoding(x)
        latent = self.initial_latent(x.shape[0])
        attns = []
        latent, attn = ca_factory()(latent, x)
        cross_attention = ca_factory()
        latent_transformer = nn.Sequential(
            [sa_factory() for _ in range(c.lt_num_blocks)]
        )
        for i in range(c.num_blocks):
            if i:
                latent, attn = cross_attention(latent, x)
            attns.append(attn)
            latent = latent_transformer(latent)
        attns = jnp.stack(attns, 1)
        latent = jnp.mean(latent, axis=-2)
        return nn.Dense(c.num_classes)(latent)

    def _positional_encoding(self, x: Array) -> Array:
        batch, h, w, d = x.shape
        pos_enc = positional_encoding(x,
                                      (-3, -2),
                                      self.config.num_freqs,
                                      (h, w))
        pos_enc = jnp.repeat(pos_enc[jnp.newaxis], batch, 0)
        x = jnp.concatenate([x, pos_enc], -1)
        pos_enc_dim = 2 * (2 * self.config.num_freqs + 1)
        x = jnp.reshape(x, (batch, h * w, d + pos_enc_dim))
        return x

    def initial_latent(self, batch_size: int | None = None) -> Array:
        shape = (self.config.latent_channels, self.config.latent_dim)
        prior = self.param('prior', nn.initializers.lecun_normal(), shape)
        if batch_size is not None:
            prior = jnp.repeat(prior[jnp.newaxis], batch_size, 0)
        return prior


class Networks(nn.Module):
    config: Config

    def setup(self) -> None:
        self.encoder = None
        self.clsf_head = None
        self.actor = None
