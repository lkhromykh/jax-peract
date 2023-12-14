import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import src.types_ as types

Array = jax.Array
activation = nn.gelu


class VoxelsProcessor(nn.Module):

    features: types.Layers
    kernels: types.Layers
    strides: types.Layers
    dtype: types.DType
    use_skip_connections: bool = True

    def setup(self) -> None:
        assert len(self.features) == len(self.kernels) == len(self.strides)
        self.convs = self._make_stem(nn.Conv)
        self.deconvs = self._make_stem(nn.ConvTranspose)

    def encode(self, x: Array) -> tuple[Array, list[Array]]:
        chex.assert_type(x, float)
        chex.assert_rank(x, 4)  # (H, W, D, C)

        skip_connections = []
        for block in self.convs:
            x = block(x)
            skip_connections.append(x)
        return x, skip_connections

    def decode(self, x: Array, skip_connections: list[Array]) -> Array:
        chex.assert_type(x, float)
        chex.assert_rank(x, 4)

        blocks_ys = zip(self.deconvs, skip_connections)
        for block, y in reversed(list(blocks_ys)):
            if self.use_skip_connections:
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
                        dtype=self.dtype,
                        use_bias=False,
                        padding='VALID',
                        )
            norm = nn.LayerNorm(dtype=self.dtype)
            block = nn.Sequential([conv, norm, activation])
            # ViT and 2106.14881 don't apply activation on the PreAttention conv.
            blocks.append(block)
        return blocks


class InputsMultiplexer(nn.Module):

    init_scale: float

    @nn.compact
    def __call__(self, *inputs: Array) -> Array:
        chex.assert_rank(inputs, 2)  # [(seq_len, channels)]
        max_dim = max(map(lambda x: x.shape[1], inputs))
        max_dim += 16 - max_dim % 8
        output = []
        for idx, val in enumerate(inputs):
            seq_len, channels = val.shape
            enc = self.param(f'modality_encoding{idx}',
                             nn.initializers.normal(self.init_scale, val.dtype),
                             (1, max_dim - channels)
                             )
            enc = jnp.repeat(enc, seq_len, 0)
            val = jnp.concatenate([val, enc], 1)
            output.append(val)
        return jnp.concatenate(output, 0)

    @staticmethod
    def inverse(input_: Array,
                shapes: list[tuple[int, ...]]
                ) -> list[Array]:
        chex.assert_rank(input_, 2)
        seq_len, channels = input_.shape
        index = 0
        outputs = []
        for shape in shapes:
            size = np.prod(shape).astype(np.int32)
            output = input_[index:index + size]
            output = output.reshape(shape + (channels,))
            outputs.append(output)
            index += size
        assert index == seq_len
        return outputs


class ActionDecoder(nn.Module):

    action_spec: types.ActionSpec
    mlp_layers: types.Layers
    conv_kernel: int
    dtype: types.DType

    @nn.compact
    def __call__(self, voxels: Array, low_dim: Array) -> tfd.Distribution:
        chex.assert_rank([voxels, low_dim], [4, 1])
        nbins = tuple(map(lambda sp: sp.num_values, self.action_spec))
        grid_size, low_dim_bins = nbins[:3], nbins[3:]
        voxels = nn.Conv(1, 3 * (self.conv_kernel,), dtype=self.dtype)(voxels)
        voxels = voxels.astype(jnp.float32)
        grid_dist = tfd.TransformedDistribution(
            distribution=tfd.Categorical(voxels.flatten()),
            bijector=ActionDecoder.Idx2Grid(grid_size)
        )
        for layer in self.mlp_layers:
            low_dim = nn.Dense(layer, dtype=self.dtype)(low_dim)
            low_dim = activation(low_dim)
        low_dim_logits = nn.Dense(sum(low_dim_bins), dtype=self.dtype)(low_dim)
        low_dim_logits = low_dim_logits.astype(jnp.float32)
        *low_dim_logits, _ = jnp.split(low_dim_logits, np.cumsum(low_dim_bins))
        low_dim_dists = [tfd.Categorical(logits) for logits in low_dim_logits]
        return ActionDecoder.Blockwise([grid_dist] + low_dim_dists)

    class Blockwise(tfd.Blockwise):

        def mode(self, *args, **kwargs):
            modes = map(lambda x: x.mode(*args, **kwargs), self.distributions)
            modes = map(jnp.atleast_1d, modes)
            return jnp.concatenate(list(modes), -1)

    class Idx2Grid(tfp.bijectors.AutoCompositeTensorBijector):

        def __init__(self, shape, validate_args=False, name='idx2grid'):
            super().__init__(validate_args=validate_args,
                             is_constant_jacobian=True,
                             forward_min_event_ndims=0,
                             inverse_min_event_ndims=1,
                             dtype=jnp.int32,
                             parameters=dict(locals()),
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
            shape = [input_shape, [len(self.shape)]]
            return np.concatenate(shape, dtype=input_shape.dtype)

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
