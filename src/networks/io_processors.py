import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import src.types_ as types

Array = jax.Array


class VoxelGridProcessor(nn.Module):

    features: types.Layers
    kernels: types.Layers
    strides: types.Layers

    def setup(self) -> None:
        self.convs = self._make_stem(nn.Conv)
        self.deconvs = self._make_stem(nn.ConvTranspose)

    # def __call__(self, *args, **kwargs):
    #     # Not for direct use, but for .init and .tabulate.
    #     return self.decode(*self.encode(*args, **kwargs))

    def encode(self, voxel_grid: Array) -> tuple[Array, list[Array]]:
        chex.assert_type(voxel_grid, jnp.uint8)
        chex.assert_rank(voxel_grid, 4)  # (H, W, D, C)

        x = voxel_grid / 128. - 1
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
            mode = map(lambda x: x.mode(*args, **kwargs), self.distributions)
            return jnp.stack(list(mode), -1)

    class Idx2Grid(tfp.bijectors.Bijector):

        def __init__(self, shape, validate_args=False, name='idx2grid'):
            super().__init__(validate_args=validate_args,
                             is_constant_jacobian=True,
                             forward_min_event_ndims=0,
                             inverse_min_event_ndims=1,
                             dtype=jnp.int32,
                             name=name)
            self.shape = shape
            self._cumprod = np.prod(shape) // np.cumprod(shape)

        @property
        def _is_permutation(self):
            return True

        def _forward(self, x):
            idxs = []
            for size in reversed(self.shape):
                x, idx = jnp.divmod(x, size)
                idxs.append(idx)
            return jnp.stack(idxs[::-1], -1)

        def _inverse(self, y):
            return jnp.matmul(y, self._cumprod)

        def _inverse_log_det_jacobian(self, y):
            return jnp.zeros([], y.dtype)

        def _forward_log_det_jacobian(self, x):
            return jnp.zeros([], x.dtype)

        def _forward_event_shape_tensor(self, input_shape):
            return np.concatenate([input_shape, [len(self.shape)]],
                                  dtype=input_shape.dtype)

        def _inverse_event_shape(self, output_shape):
            return output_shape[:-1]

        def _forward_event_shape(self, input_shape):
            return input_shape + (len(self.shape),)

        def _inverse_event_shape(self, output_shape):
            return output_shape[:-1]

    @nn.compact
    def __call__(self, vgrid: Array, low_dim: Array) -> tfd.Distribution:
        chex.assert_rank([vgrid, low_dim], [4, 1])  # (seq_len, channels)

        vgrid_dist = self._decode_vgrid(vgrid)
        low_dim_dists = self._decode_low_dim(low_dim)
        return ActionDecoder.Blockwise([vgrid_dist] + low_dim_dists)

    def _decode_vgrid(self, vgrid: Array) -> tfd.Distribution:
        vgrid = nn.Conv(1, (1, 1, 1))(vgrid)
        grid_size = tuple(map(lambda sp: sp.num_values, self.act_spec[:3]))
        return tfd.TransformedDistribution(
            distribution=tfd.Categorical(vgrid.flatten()),
            bijector=ActionDecoder.Idx2Grid(grid_size)
        )

    def _decode_low_dim(self, x: Array) -> list[tfd.Distribution]:
        nbins = tuple(map(lambda sp: sp.num_values, self.act_spec[3:]))
        low_dim_logits = nn.Dense(sum(nbins))(x)
        *low_dim_logits, _ = jnp.split(low_dim_logits, np.cumsum(nbins))
        return [tfd.Categorical(logits) for logits in low_dim_logits]
