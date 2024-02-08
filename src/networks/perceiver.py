"""PerceiverIO"""
from typing import Any

import jax
import jax.numpy as jnp
import chex
import numpy as np
import flax.linen as nn

Array = jax.Array
DType = Any


class _Module(nn.Module):

    dtype: DType
    kernel_init: nn.initializers.Initializer
    use_layernorm: bool

    def dense(self, x: Array, **kwargs) -> Array:
        return nn.DenseGeneral(dtype=self.dtype,
                               kernel_init=self.kernel_init,
                               use_bias=True,
                               **kwargs
                               )(x)

    def norm(self, x: Array, **kwargs) -> Array:
        if self.use_layernorm:
            return nn.LayerNorm(use_bias=True,
                                use_scale=True,
                                dtype=self.dtype,
                                **kwargs)(x)
        return x


def geglu(x: Array) -> Array:
    x, gates = jnp.split(x, 2, -1)
    return x * nn.gelu(gates)


class MLP(_Module):

    widening_factor: float

    @nn.compact
    def __call__(self, x: Array) -> Array:
        dim = x.shape[-1]
        x = self.dense(x, features=2 * int(self.widening_factor * dim))
        x = geglu(x)
        return self.dense(x, features=dim)


def scaled_dot_product(query: Array,
                       key: Array,
                       value: Array,
                       ) -> Array:
    attn = jnp.einsum('...qhd,...khd->...hqk', query, key)
    attn /= np.sqrt(query.shape[-1])
    attn = jax.nn.softmax(attn)
    return jnp.einsum('...hqk,...khd->...qhd', attn, value)


class MultiHeadAttention(_Module):

    num_heads: int
    qk_channels: int | None = None
    v_channels: int | None = None
    output_channels: int | None = None

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 ) -> Array:
        def mh_dense(x, dim, name):
            dim, res = np.divmod(dim, self.num_heads)
            assert res == 0, f'Not divisible by the number of heads: {dim}.'
            return self.dense(x,
                              features=(self.num_heads, dim),
                              name=name)
        qk_channels = self.qk_channels or inputs_q.shape[-1]
        v_channels = self.v_channels or qk_channels
        output_channels = self.output_channels or v_channels

        query = mh_dense(inputs_q, qk_channels, 'query')
        key = mh_dense(inputs_kv, qk_channels, 'key')
        value = mh_dense(inputs_kv, v_channels, 'value')
        value = scaled_dot_product(query, key, value)
        return self.dense(value,
                          features=output_channels,
                          axis=(-2, -1),
                          name='proj')


class CrossAttention(_Module):

    num_heads: int
    widening_factor: float
    use_query_residual: bool = True

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array | None = None,
                 ) -> Array:
        ln_inputs_q = self.norm(inputs_q)
        if inputs_kv is not None:
            inputs_kv = self.norm(inputs_kv)  # cross-attention
        else:
            inputs_kv = ln_inputs_q  # self-attention
        x = MultiHeadAttention(
            num_heads=self.num_heads,
            qk_channels=inputs_kv.shape[-1],
            output_channels=inputs_q.shape[-1],
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            use_layernorm=self.use_layernorm
        )(ln_inputs_q, inputs_kv)
        if self.use_query_residual:
            x += inputs_q
        mlp = MLP(
            widening_factor=self.widening_factor,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            use_layernorm=self.use_layernorm
        )
        return x + mlp(self.norm(x))


# TODO: propagate attention all through.
class PerceiverIO(nn.Module):

    latent_dim: int
    latent_channels: int
    num_blocks: int
    num_self_attend_per_block: int
    num_cross_attend_heads: int
    num_self_attend_heads: int
    cross_attend_widening_factor: float
    self_attend_widening_factor: float
    use_decoder_query_residual: bool = False
    prior_initial_scale: float = 0.02
    dtype: DType = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    use_layernorm: bool = True

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 outputs_q: Array
                 ) -> Array:
        chex.assert_type([inputs_q, outputs_q], float)
        chex.assert_rank([inputs_q, outputs_q], 2)  # (seq_len, channels)
        encode_query = CrossAttention(
            num_heads=self.num_cross_attend_heads,
            widening_factor=self.cross_attend_widening_factor,
            use_query_residual=True,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            use_layernorm=self.use_layernorm
        )
        decode_query = CrossAttention(
            num_heads=self.num_cross_attend_heads,
            widening_factor=self.cross_attend_widening_factor,
            use_query_residual=self.use_decoder_query_residual,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            use_layernorm=self.use_layernorm
        )
        latent_transformer = nn.Sequential([
            CrossAttention(
                num_heads=self.num_self_attend_heads,
                widening_factor=self.self_attend_widening_factor,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                use_layernorm=self.use_layernorm
            )
            for _ in range(self.num_self_attend_per_block)
        ])

        latent = self.param(
            'latent_prior',
            nn.initializers.normal(self.prior_initial_scale),
            (self.latent_dim, self.latent_channels)
        )
        latent = encode_query(latent, inputs_q)
        for _ in range(self.num_blocks):
            latent = latent_transformer(latent)
        return decode_query(outputs_q, latent)
