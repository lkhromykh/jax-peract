import jax.numpy as jnp
import flax.linen as nn
import chex
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from src import utils
import src.types_ as types
from src.config import Config
from src.networks import io_processors
from src.networks.perceiver import PerceiverIO


class PerAct(nn.Module):

    config: Config
    action_spec: types.ActionSpec

    def setup(self) -> None:
        c = self.config
        dtype = _dtype_fromstr(c.compute_dtype)
        self.voxels_proc = io_processors.VoxelsProcessor(
            features=c.conv_stem_features,
            kernels=c.conv_stem_kernels,
            strides=c.conv_stem_strides,
            patch_size=c.voxels_patch_size,
            dtype=dtype,
            kernel_init=nn.initializers.lecun_normal(),
            use_skip_connections=c.conv_stem_use_skip_connections,
        )
        self.perceiver = PerceiverIO(
            latent_dim=c.latent_dim,
            latent_channels=c.latent_channels,
            num_blocks=c.num_blocks,
            num_self_attend_per_block=c.num_self_attend_per_block,
            num_cross_attend_heads=c.num_cross_attend_heads,
            num_self_attend_heads=c.num_self_attend_heads,
            cross_attend_widening_factor=c.cross_attend_widening_factor,
            self_attend_widening_factor=c.self_attend_widening_factor,
            prior_initial_scale=c.prior_initial_scale,
            dtype=dtype,
            kernel_init=nn.initializers.lecun_normal(),
            use_layer_norm=c.use_layer_norm
        )
        self.action_decoder = io_processors.ActionDecoder(
            action_spec=self.action_spec,
            mlp_dim=c.act_decoder_mlp_dim,
            conv_kernel=c.act_decoder_conv_kernel,
            dtype=jnp.float32,
            kernel_init=nn.initializers.normal(1e-2)
        )

    @nn.compact
    def __call__(self, obs: types.State) -> tfd.Distribution:
        chex.assert_rank([obs.voxels, obs.low_dim, obs.goal], [4, 1, 2])
        chex.assert_type([obs.voxels, obs.low_dim, obs.goal], [jnp.uint8, float, float])
        c = self.config
        dtype = _dtype_fromstr(c.compute_dtype)
        voxels, low_dim, task = map(lambda x: x.astype(dtype), obs)
        voxels = voxels / dtype(128) - 1
        patches, skip_connections = self.voxels_proc.encode(voxels)
        patches_shape = patches.shape[:-1]

        def tokens_preproc(x, name):
            x = x.reshape(-1, x.shape[-1])
            fc = nn.Dense(c.tokens_dim, use_bias=False, dtype=dtype, name=f'{name}_tokens_dense')
            ln = nn.LayerNorm(dtype=dtype, name=f'{name}_tokens_ln')
            return ln(fc(x))
        patches = tokens_preproc(patches, 'voxels')
        low_dim = tokens_preproc(low_dim, 'low_dim')
        task = tokens_preproc(task, 'task')
        pos1d_enc = utils.fourier_features(task.shape[:1], c.ff_num_bands).astype(dtype)
        pos3d_enc = utils.fourier_features(patches_shape, c.ff_num_bands).astype(dtype)
        pos3d_enc = pos3d_enc.reshape(-1, pos3d_enc.shape[-1])
        patches = jnp.concatenate([patches, pos3d_enc], -1)
        task = jnp.concatenate([task, pos1d_enc], -1)
        inputs_q = io_processors.InputsMultiplexer(c.prior_initial_scale)(
            patches, low_dim, task
        )
        outputs_q = io_processors.InputsMultiplexer(c.prior_initial_scale)(
            patches, low_dim
        )
        outputs_val = self.perceiver(inputs_q, outputs_q)
        outputs_val = nn.LayerNorm(dtype=dtype, name='representation_ln')(outputs_val)
        patches, low_dim = io_processors.InputsMultiplexer.inverse(
            outputs_val, shapes=[patches_shape, ()]
        )
        voxels = self.voxels_proc.decode(patches, skip_connections)
        chex.assert_type([inputs_q, outputs_q, voxels], dtype)
        return self.action_decoder(voxels, low_dim)


def _dtype_fromstr(dtype_str: str) -> types.DType:
    valid_dtypes = dict(bf16=jnp.bfloat16, f32=jnp.float32)
    return valid_dtypes[dtype_str]
