import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import src.types_ as types

Array = jax.Array


class VoxelsProcessor(nn.Module):

    features: types.Layers
    kernels: types.Layers
    strides: types.Layers

    def setup(self) -> None:
        self.convs = self._make_stem(nn.Conv)
        self.deconvs = self._make_stem(nn.ConvTranspose)

    # def __call__(self, *args, **kwargs):
    #     # Not for direct use, but for .init and .tabulate.
    #     return self.decode(*self.encode(*args, **kwargs))

    def encode(self, voxels: Array) -> tuple[Array, list[Array]]:
        chex.assert_type(voxels, jnp.uint8)
        chex.assert_rank(voxels, 4)  # (H, W, D, C)

        x = voxels / 128. - 1
        skip_connections = []
        for block in self.convs:
            x = block(x)
            skip_connections.append(x)
        return x, skip_connections

    def decode(self, x: Array, skip_connections: list[Array]) -> Array:
        chex.assert_type(x, float)
        chex.assert_rank(x, 4)

        blocks_ys = list(zip(self.deconvs, skip_connections))
        for block, y in reversed(blocks_ys):
            x = jnp.concatenate([x, y], -1)
            x = block(x)
        return x

    def _make_stem(self, Conv: nn.Conv | nn.ConvTranspose) -> nn.Sequential:
        blocks = []
        arch = zip(self.features, self.kernels, self.strides)
        for f, k, s in arch:
            conv = Conv(features=f,
                        kernel_size=3 * (k,),
                        strides=3 * (s,),
                        use_bias=False,
                        padding='VALID'
                        )
            block = nn.Sequential([conv, nn.gelu])
            blocks.append(block)
        return blocks


class InputsMultiplexer(nn.Module):

    init_scale: float

    @nn.compact
    def __call__(self, *inputs: Array) -> Array:
        chex.assert_rank(inputs, 2)  # [(seq_len, channels)]
        max_dim = max(map(lambda x: x.shape[-1], inputs))
        max_dim += 8 - max_dim % 4
        outputs = []
        for idx, val in enumerate(inputs):
            seq_len, channels = val.shape
            enc = self.param(f'encoding_{idx}',
                             nn.initializers.normal(self.init_scale),
                             (1, max_dim - channels)
                             )
            enc = jnp.repeat(enc, seq_len, 0)
            val = jnp.concatenate([val, enc], 1)
            outputs.append(val)
        return jnp.concatenate(outputs, 0)


class ActionDecoder(nn.Module):

    act_spec: types.ActionSpec

    class Blockwise(tfd.Blockwise):
        def mode(self, *args, **kwargs):
            modes = map(lambda x: x.mode(*args, **kwargs), self.distributions)
            modes = map(jnp.atleast_1d, modes)
            return jnp.concatenate(list(modes), -1)

    class Idx2Grid(tfp.bijectors.AutoCompositeTensorBijector):

        def __init__(self, shape, validate_args=True, name='idx2grid'):
            super().__init__(validate_args=validate_args,
                             is_constant_jacobian=True,
                             forward_min_event_ndims=0,
                             inverse_min_event_ndims=1,
                             dtype=jnp.int32,
                             parameters=dict(shape=shape),
                             name=name)
            self.shape = shape

        @property
        def _is_permutation(self):
            return True

        def _forward(self, x):
            idxs = jnp.unravel_index(x, self.shape)
            return jnp.stack(idxs, -1)

        def _inverse(self, y):
            return jnp.ravel_multi_index(y, self.shape, mode='clip')

        def _inverse_log_det_jacobian(self, y):
            return jnp.zeros([], y.dtype)

        def _forward_event_shape_tensor(self, input_shape):
            return np.concatenate([input_shape, [len(self.shape)]],
                                  dtype=self.dtype)

        def _inverse_event_shape_tensor(self, output_shape):
            return output_shape[:-1]

        def _forward_event_shape(self, input_shape):
            return input_shape + (len(self.shape),)

        def _inverse_event_shape(self, output_shape):
            return output_shape[:-1]

        def _forward_dtype(self, input_dtype, **kwargs):
            return self.dtype

        def _inverse_dtype(self, output_dtype, **kwargs):
            return self.dtype

    @nn.compact
    def __call__(self, voxels: Array, low_dim: Array) -> tfd.Distribution:
        chex.assert_rank([voxels, low_dim], [4, 1])  # (seq_len, channels)
        nbins = tuple(map(lambda sp: sp.num_values, self.act_spec))
        grid_size, low_dim_bins = nbins[:3], nbins[3:]
        voxels = nn.Conv(1, (1, 1, 1))(voxels)
        grid_dist = tfd.TransformedDistribution(
            distribution=tfd.Categorical(voxels.flatten()),
            bijector=ActionDecoder.Idx2Grid(grid_size)
        )
        low_dim_logits = nn.Dense(sum(low_dim_bins))(low_dim)
        *low_dim_logits, _ = jnp.split(low_dim_logits, np.cumsum(low_dim_bins))
        low_dim_dists = [tfd.Categorical(logits) for logits in low_dim_logits]
        return ActionDecoder.Blockwise(
            distributions=[grid_dist] + low_dim_dists,
            dtype_override=jnp.int32
        )
