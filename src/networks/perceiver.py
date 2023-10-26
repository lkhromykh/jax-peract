import jax
import jax.numpy as jnp
import chex
import numpy as np
import flax.linen as nn

Array = jax.Array
_dki = nn.initializers.variance_scaling(
    scale=.4,
    mode='fan_in',
    distribution='truncated_normal')


def norm(x: Array) -> Array:
    ln = nn.LayerNorm(use_bias=True, use_scale=True)
    return ln(x)


class MLP(nn.Module):

    widening_factor: float

    @nn.compact
    def __call__(self, x: Array) -> Array:
        dim = x.shape[-1]
        x = nn.Dense(int(self.widening_factor * dim), kernel_init=_dki)(x)
        x = nn.gelu(x)
        return nn.Dense(dim, kernel_init=_dki)(x)


def scaled_dot_product(query: Array,
                       key: Array,
                       value: Array,
                       ) -> Array:
    attn = jnp.einsum('...qhd,...khd->...hqk', query, key)
    attn /= np.sqrt(query.shape[-1])
    attn = jax.nn.softmax(attn)
    return jnp.einsum('...hqk,...khd->...qhd', attn, value)


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
    widening_factor: float
    use_residual: bool = True

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 ) -> Array:
        channels = inputs_q.shape[-1]
        mha = MultiHeadAttention(self.num_heads,
                                 qk_channels=channels,
                                 output_channels=channels)
        x = mha(norm(inputs_q), norm(inputs_kv))
        if self.use_residual:
            x += inputs_q
        mlp = MLP(self.widening_factor)
        return x + mlp(norm(x))


class SelfAttention(nn.Module):

    num_heads: int
    widening_factor: float

    @nn.compact
    def __call__(self, x: Array) -> Array:
        qkv = norm(x)
        mha = MultiHeadAttention(self.num_heads)
        x += mha(qkv, qkv)
        mlp = MLP(self.widening_factor)
        return x + mlp(norm(x))


class PerceiverIO(nn.Module):

    latent_dim: int
    latent_channels: int
    num_blocks: int
    num_self_attend_per_block: int
    num_cross_attend_heads: int
    num_self_attend_heads: int
    cross_attend_widening_factor: float
    self_attend_widening_factor: float
    use_query_residual: bool = True
    prior_initial_scale: float = 0.02

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 outputs_q: Array
                 ) -> Array:
        chex.assert_type([inputs_q, outputs_q], float)
        chex.assert_rank([inputs_q, outputs_q], 2)  # (seq_len, channels)

        encoder_query = CrossAttention(self.num_cross_attend_heads,
                                       self.cross_attend_widening_factor)
        decoder_query = CrossAttention(self.num_cross_attend_heads,
                                       self.cross_attend_widening_factor,
                                       self.use_query_residual)
        latent_transformer = nn.Sequential([
            SelfAttention(self.num_self_attend_heads,
                          self.self_attend_widening_factor)
            for _ in range(self.num_self_attend_per_block)
        ])

        latent = self.param(
            'prior',
            nn.initializers.normal(self.prior_initial_scale),
            (self.latent_dim, self.latent_channels)
        )
        latent = encoder_query(latent, inputs_q)
        for _ in range(self.num_blocks):
            latent = latent_transformer(latent)
        return decoder_query(outputs_q, latent)
