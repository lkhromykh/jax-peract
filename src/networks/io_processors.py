from typing import Type, Literal

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import src.types_ as types
from src.utils import distributions

Array = jax.Array
activation = nn.gelu


class VoxelsProcessor(nn.Module):
    """Process voxel grid with 3D convolutions."""

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
        chex.assert_rank(x, 4)  # (X, Y, Z, C)

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

    def _make_stem(self, conv_cls: Type[nn.Conv] | Type[nn.ConvTranspose]) -> list[nn.Module]:
        blocks = []
        arch = list(zip(self.features, self.kernels, self.strides))
        for i, (f, k, s) in enumerate(arch):
            no_layernorm = i == len(arch) - 1 and conv_cls == nn.Conv
            # The authors of 2106.14881 don't apply activation on the pre-transformer conv.
            conv = conv_cls(features=f,
                            kernel_size=3 * (k,),
                            strides=3 * (s,),
                            dtype=self.dtype,
                            use_bias=no_layernorm,
                            padding='VALID')
            block = [conv] if no_layernorm else [conv, nn.LayerNorm(dtype=self.dtype), activation]
            blocks.append(nn.Sequential(block))
        return blocks


class InputsMultiplexer(nn.Module):
    """Concatenate/split modalities."""

    init_scale: float
    pad_to: int = 4

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


# try conditioned low-dim decoding via tfp.JointDistributionSequential?
class ActionDecoder(nn.Module):
    """Policy decoder."""

    action_spec: types.ActionSpec
    mlp_dim: types.Layers
    conv_kernel: int
    dtype: types.DType
    mode: Literal['old', 'independent', 'auto_regressive'] = 'autoregressive'

    def __call__(self, voxels: Array, low_dim: Array) -> tfd.Distribution:
        vgrid_logits, low_dim_logits = self._nn_process(voxels, low_dim)
        match self.mode:
            case 'old': dist_fn = self._old_policy
            case 'independent': dist_fn = self._independent_policy
            case 'autoregressive': dist_fn = self._autoregressive_policy
            case _: raise ValueError(self.mode)
        return dist_fn(vgrid_logits, low_dim_logits)

    @nn.compact
    def _nn_process(self, voxels: Array, low_dim: Array) -> tuple[Array, Array]:
        chex.assert_rank([voxels, low_dim], [4, 1])
        _, _, _, vgrid_channels, low_dim_dof = self._infer_shapes()
        conv = nn.Conv(vgrid_channels, 3 * (self.conv_kernel,), dtype=self.dtype, name='vgrid_logits')
        vgrid_logits = conv(voxels).astype(jnp.float32)

        low_dim = nn.Dense(self.mlp_dim, dtype=self.dtype, name='low_dim_hidden')(low_dim)
        low_dim = activation(low_dim)
        low_dim_logits = nn.Dense(low_dim_dof, dtype=self.dtype, name='low_dim_logits')(low_dim)
        return vgrid_logits, low_dim_logits

    def _old_policy(self, vgrid_logits: Array, low_dim_logits: Array) -> tfd.Distribution:
        grid_size, gripper_dof, low_dim_dof, _, _ = self._infer_shapes()
        grid_dist = tfd.TransformedDistribution(
            distribution=tfd.Categorical(vgrid_logits.flatten()),
            bijector=distributions.Idx2Grid(grid_size),
            name='grid_distribution'
        )
        low_dim_dof = gripper_dof + low_dim_dof
        *low_dim_logits, _ = jnp.split(low_dim_logits, np.cumsum(low_dim_dof))
        low_dim_dists = [tfd.Categorical(logits) for logits in low_dim_logits]
        return distributions.Blockwise([grid_dist] + low_dim_dists)

    def _independent_policy(self, vgrid_logits: Array, low_dim_logits: Array) -> tfd.Distribution:
        grid_size, gripper_dof, low_dim_dof, _, _ = self._infer_shapes()
        grid_dist = tfd.TransformedDistribution(
            distribution=tfd.Categorical(vgrid_logits.flatten()),
            bijector=distributions.Idx2Grid(grid_size),
            name='grid_distribution'
        )
        low_dim_dof = gripper_dof + low_dim_dof
        *low_dim_logits, _ = jnp.split(low_dim_logits, np.cumsum(low_dim_dof))
        low_dim_dists = [tfd.Categorical(logits) for logits in low_dim_logits]
        return distributions.JointDistributionSequential(
            [grid_dist] + low_dim_dists,
            batch_ndims=0, validate_args=True
        )

    def _autoregressive_policy(self, vgrid_logits: Array, low_dim_logits: Array) -> tfd.Distribution:
        grid_size, gripper_dof, low_dim_dof, _, _ = self._infer_shapes()
        vgrid_logits, vgrid_gripper_logits = jnp.split(vgrid_logits, [1], -1)
        grid_dist = tfd.TransformedDistribution(
            distribution=tfd.Categorical(vgrid_logits.flatten()),
            bijector=distributions.Idx2Grid(grid_size),
            name='grid_distribution'
        )

        # def gripper_dist_fn(grid_distribution):
        #     *gripper_logits, _ = jnp.split(vgrid_gripper_logits[tuple(grid_distribution)], np.cumsum(gripper_dof), -1)
        #     dists = [tfd.Categorical(logits) for logits in gripper_logits]
        #     return distributions.Blockwise(dists)

        def gripper_dist_fn(dist_idx):
            def dist_fn(*args):
                import pdb; pdb.set_trace()
                grid_idx = tuple(args[-1])
                *gripper_logits, _ = jnp.split(vgrid_gripper_logits[grid_idx], np.cumsum(gripper_dof), -1)
                return tfd.Categorical(gripper_logits[dist_idx])
            return dist_fn

        gripper_dist_fns = [gripper_dist_fn(i) for i, _ in enumerate(gripper_dof)]

        *low_dim_logits, _ = jnp.split(low_dim_logits, np.cumsum(low_dim_dof))
        low_dim_dists = [tfd.Categorical(logits) for logits in low_dim_logits]
        return distributions.JointDistributionSequential(
            [grid_dist] + gripper_dist_fns + low_dim_dists,
            batch_ndims=0, validate_args=True
        )

    def _infer_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int,...], int, int]:
        nbins = tuple(map(lambda sp: sp.num_values, self.action_spec))
        grid_size, gripper_dof, global_dof = nbins[:3], nbins[3:7], nbins[7:]
        match self.mode:
            case 'old' | 'independent':
                vgrid_channels = 1
                low_dim_channels = sum(gripper_dof + global_dof)
            case 'autoregressive':
                vgrid_channels = 1 + sum(gripper_dof)
                low_dim_channels = sum(global_dof)
            case _:
                raise ValueError(self.mode)
        return grid_size, gripper_dof, global_dof, vgrid_channels, low_dim_channels
