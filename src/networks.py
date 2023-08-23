import functools
from typing import Callable, Any

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from src.positional_encoding import positional_encoding

Layers = Any


def get_act(act: str) -> Any:
    if act == 'none':
        return lambda x: x
    if hasattr(jax.nn, act):
        return getattr(jax.nn, act)
    if hasattr(nn, act):
        return getattr(nn, act)
    raise NotImplementedError


def get_norm(norm: str) -> Any:
    if norm == 'none':
        return lambda x: x
    if norm == 'layer':
        return nn.LayerNorm()
    raise NotImplementedError


def scaled_dot_product(query, key, value, mask=None):
    # todo: time_major
    attn = jnp.einsum('...qhd,...khd->...hqk', query, key)
    attn /= np.sqrt(query.shape[-1])
    attn = jax.nn.softmax(attn)
    return jnp.einsum('...hqk,...khd->...qhd', attn, value), attn


class MLP(nn.Module):

    layers: Layers
    activation: str
    normalization: str
    activate_final: bool = True

    @nn.compact
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.Dense(layer)(x)
            if i != len(self.layers) - 1 or self.activate_final:
                x = get_norm(self.normalization)(x)
                x = get_act(self.activation)(x)
        return x


class MultiHeadAttention(nn.Module):

    num_heads: int
    embed_dim: int

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, mask=None):
        dense = functools.partial(
            nn.DenseGeneral,
            features=(self.num_heads, self.embed_dim)
        )
        query = dense(name='query')(inputs_q)
        key = dense(name='key')(inputs_kv)
        value = dense(name='value')(inputs_kv)
        value, attn = scaled_dot_product(query, key, value, mask)
        proj = nn.DenseGeneral(self.embed_dim, axis=(-2, -1), name='proj')
        return proj(value), attn


class CrossAttention(nn.Module):
    ...


class TransformerBlock(nn.Module):

    dmodel: int
    num_heads: int
    dim_feedforward: int
    activation: str

    @nn.compact
    def __call__(self, x, mask=None):
        xln = nn.LayerNorm()(x)
        mha = MultiHeadAttention(self.num_heads, self.dmodel)
        val, attn = mha(xln, xln, mask)
        x = x + val
        xln = nn.LayerNorm()(x)
        mlp = MLP(layers=(self.dim_feedforward, self.dmodel),
                  activation=self.activation,
                  normalization='none',
                  activate_final=False)
        val = mlp(xln)
        return x + val, attn


class TransformerEncoder(nn.Module):

    dmodel: int
    num_layers: int
    num_heads: int
    dim_feedforward: int
    activation: str

    @nn.compact
    def __call__(self, x, mask=None):
        attns = []
        for _ in range(self.num_layers):
            layer = TransformerBlock(
                self.dmodel,
                self.num_heads,
                self.dim_feedforward,
                self.activation
            )
            x, attn = layer(x)
            attns.append(attn)
        return x, jnp.stack(attns)


class Networks(nn.Module):

    config: Any
    num_classes: int = 10

    def setup(self):
        c = self.config
        self.encoder = TransformerEncoder(
            c.latent_dim,
            c.num_layers,
            c.num_heads,
            c.dim_feedforward,
            c.activation
        )
        self.proj = nn.Dense(self.encoder.dmodel)
        self.clsf_head = nn.Dense(self.num_classes)

    def __call__(self, x):
        batch, h, w, c = x.shape
        pos_enc = positional_encoding(x,
                                      (-3, -2),
                                      self.config.num_freqs,
                                      self.config.nyquist_freq)
        pos_enc = jnp.repeat(pos_enc[jnp.newaxis], batch, 0)
        x = jnp.concatenate([x, pos_enc], -1)
        pos_enc_dim = 2 * (2 * self.config.num_freqs + 1)
        x = jnp.reshape(x, (batch, h * w, c + pos_enc_dim))
        x = self.proj(x)
        emb, _ = self.encoder(x)
        emb = jnp.reshape(emb, (batch, h * w * self.encoder.dmodel))
        return self.clsf_head(emb)
