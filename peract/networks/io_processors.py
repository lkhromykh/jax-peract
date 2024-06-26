from typing import Type

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import peract.types_ as types
from peract.utils import distributions

Array = jax.Array
activation = nn.gelu


class VoxelsProcessor(nn.Module):
    """Process voxel grid with 3D convolutions."""

    features: types.Layers
    kernels: types.Layers
    strides: types.Layers
    patch_size: int
    dtype: types.DType
    kernel_init: nn.initializers.Initializer
    use_skip_connections: bool = True

    def setup(self) -> None:
        assert len(self.features) == len(self.kernels) == len(self.strides)
        self.convs = self._make_stem(nn.Conv)
        self.deconvs = self._make_stem(nn.ConvTranspose)

    def encode(self, voxel_grid: Array) -> tuple[Array, list[Array]]:
        """Preprocess and extract patches."""
        chex.assert_type(voxel_grid, float)
        chex.assert_rank(voxel_grid, 4)  # (X, Y, Z, C)

        x = voxel_grid
        skip_connections = []
        for block in self.convs:
            x = block(x)
            skip_connections.append(x)

        if self.patch_size > 1:
            assert x.shape[0] % self.patch_size == 0, f'{x.shape} / {self.patch_size}'
            window_strides = 3 * (self.patch_size,)
            patches = jax.lax.conv_general_dilated_patches(
                x[jnp.newaxis],
                filter_shape=window_strides,
                window_strides=window_strides,
                padding='VALID',
                dimension_numbers=('NXYZC', 'XYZIO', 'NXYZC'),
                precision=jax.lax.Precision.DEFAULT
            ).squeeze(0)
        else:
            patches = x
        return patches, skip_connections

    def decode(self, patches: Array, skip_connections: list[Array]) -> Array:
        """Restore voxel grid from patches and postprocess."""
        chex.assert_type(patches, float)
        chex.assert_rank(patches, 4)

        if self.patch_size > 1:
            shape = 3 * (self.patch_size * patches.shape[0],) + (patches.shape[-1],)
            x = jax.image.resize(patches, shape, method='trilinear', precision=jax.lax.Precision.DEFAULT)
        else:
            x = patches

        blocks_ys = zip(self.deconvs, skip_connections)
        for block, y in reversed(list(blocks_ys)):
            if self.use_skip_connections:
                x = jnp.concatenate([x, y], -1)
            x = block(x)
        return x

    def _make_stem(self, conv_cls: Type[nn.Conv] | Type[nn.ConvTranspose]) -> list[nn.Module]:
        blocks = []
        for f, k, s in zip(self.features, self.kernels, self.strides):
            conv = conv_cls(features=f,
                            kernel_size=3 * (k,),
                            strides=3 * (s,),
                            dtype=self.dtype,
                            kernel_init=self.kernel_init,
                            use_bias=False,
                            padding='VALID')
            blocks.append(nn.Sequential([conv, nn.LayerNorm(dtype=self.dtype), activation]))
        return blocks


class InputsMultiplexer(nn.Module):
    """Concatenate/split modalities."""

    init_scale: float
    pad_to: int = 8

    @nn.compact
    def __call__(self, *inputs: Array) -> Array:
        chex.assert_rank(inputs, 2)  # [(seq_len, channels)]
        max_dim = max(map(lambda x: x.shape[1], inputs))
        max_dim += 2 * self.pad_to - max_dim % self.pad_to
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
        assert index == seq_len, "Requested shapes don't match input size."
        return outputs


class ActionDecoder(nn.Module):
    """Policy decoder."""

    action_spec: types.ActionSpec
    mlp_dim: types.Layers
    conv_kernel: int
    dtype: types.DType
    kernel_init: nn.initializers.Initializer

    @nn.compact
    def __call__(self, voxels: Array, low_dim: Array) -> tfd.Distribution:
        chex.assert_rank([voxels, low_dim], [4, 1])
        nbins = tuple(map(lambda sp: sp.num_values, self.action_spec))
        grid_size, low_dim_dof = nbins[:3], nbins[3:]

        conv = nn.Conv(
            features=1,
            kernel_size=3 * (self.conv_kernel,),
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='vgrid_logits'
        )
        vgrid_logits = conv(voxels).astype(jnp.float32)
        low_dim = nn.Dense(self.mlp_dim, dtype=self.dtype, name='low_dim_hidden')(low_dim)
        low_dim = activation(low_dim)
        low_dim_logits = nn.Dense(
            features=sum(low_dim_dof),
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            name='low_dim_logits'
        )(low_dim)
        grid_dist = tfd.TransformedDistribution(
            distribution=tfd.Categorical(vgrid_logits.flatten()),
            bijector=distributions.Idx2Grid(grid_size),
            name='voxel_grid_logits'
        )
        *low_dim_logits, _ = jnp.split(low_dim_logits, np.cumsum(low_dim_dof))
        low_dim_dists = [tfd.Categorical(logits) for logits in low_dim_logits]
        dist = distributions.Blockwise([grid_dist] + low_dim_dists)
        if self.is_initializing():
            # Specifically for nn.tabulate since tfd.Distribution is interfering with jax.eval_shape.
            return dist.mode()
        return dist
