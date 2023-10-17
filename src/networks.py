import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
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
        obs = self._maybe_batch(obs)
        known_modalities = {
            # 'voxels': self._convs,
            'low_dim': self._mlp,
            # 'task': self._mlp
        }
        out = {}
        breakpoint()
        for key, fn in known_modalities.items():
            out[key] = fn(obs[key])
        for key, val in ops.multimodal_encoding(out).items():
            out[key] = jnp.concatenate([out[key], val], -1)
        return jnp.concatenate(list(out.values()), 1)

    def _convs(self, voxel_grid: Array) -> Array:
        x = voxel_grid / 255.
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


class ActPreprocess(nn.Module):

    act_spec: types.ActionSpec

    class Blockwise(tfd.Blockwise):
        def mode(self, *args, **kwargs):
            mode = map(lambda x: x.mode(*args, **kwargs), self.distributions)
            return jnp.stack(list(mode), -1)

    @nn.compact
    def __call__(self, state: Array) -> tfd.Distribution:
        nbins = list(map(lambda sp: sp.num_values, self.act_spec))
        logits = nn.Dense(sum(nbins))(state)
        *logits, _ = jnp.split(logits, np.cumsum(nbins), -1)
        return ActPreprocess.Blockwise([tfd.Categorical(log) for log in logits])


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
        self.postprocessor = ActPreprocess(self.act_spec)

    @nn.compact
    def __call__(self, obs: types.Observation) -> tfd.Distribution:
        x = MLP(512)(jnp.atleast_2d(obs['low_dim']))
        # x = self.preprocessor(obs)
        # x = self.encoder(x)
        return self.postprocessor(x)


class Perceiver(nn.Module):
    config: Config
    observation_spec: types.ObservationSpec
    action_spec: types.ActionSpec

    def __call__(self, obs: types.Observation) -> tfd.Distribution:
        """God-module required if plan to use skip-connection"""
