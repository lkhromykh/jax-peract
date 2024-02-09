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


# TODO: properly name all encodings/priors to apply weight decay.
#  name dense modules to distinguish them.
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
            dtype=dtype,
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
            use_decoder_query_residual=c.use_decoder_query_residual,
            prior_initial_scale=c.prior_initial_scale,
            dtype=dtype,
            kernel_init=nn.initializers.lecun_normal(),
            use_layernorm=c.use_layernorm
        )
        self.action_decoder = io_processors.ActionDecoder(
            action_spec=self.action_spec,
            mlp_layers=c.act_decoder_mlp_layers,
            conv_kernel=c.act_decoder_conv_kernel,
            dtype=dtype
        )

    @nn.compact
    def __call__(self, obs: types.State) -> tfd.Distribution:
        chex.assert_rank([obs.voxels, obs.low_dim, obs.goal], [4, 1, 2])
        chex.assert_type([obs.voxels, obs.low_dim], [jnp.uint8, float])
        c = self.config
        dtype = _dtype_fromstr(c.compute_dtype)
        voxels, low_dim, task = map(lambda x: x.astype(dtype), obs)
        voxels = voxels / 128. - 1
        patches, skip_connections = self.voxels_proc.encode(voxels)
        patches_shape, channels = patches.shape[:3], patches.shape[-1]
        pos3d_enc = utils.fourier_features(patches_shape, c.ff_num_bands)
        if c.use_trainable_pos_encoding:  # Hide 3D structure of the voxels.
            pos3d_enc = self.param(
                'input_pos3d_encoding',
                nn.initializers.normal(c.prior_initial_scale, patches.dtype),
                pos3d_enc.shape  # Make further capacity equivalent.
            )
        patches = jnp.concatenate([patches, pos3d_enc], -1)
        patches = patches.reshape(-1, patches.shape[-1])
        low_dim = nn.Dense(channels, dtype=dtype)(low_dim).reshape(1, -1)
        task = nn.Dense(channels, dtype=dtype)(task)
        pos1d_enc = utils.fourier_features(task.shape[:1], c.ff_num_bands)
        task = jnp.concatenate([task, pos1d_enc], -1)

        inputs_q = io_processors.InputsMultiplexer(c.prior_initial_scale)(
            patches, low_dim, task
        )
        # TODO: try different options for decoder query
        if c.use_trainable_pos_encoding:
            patches = self.param(
                'output_pos3d_encoding',
                nn.initializers.normal(c.prior_initial_scale, patches.dtype),
                patches.shape
            )
        low_dim = self.param(
            'low_dim_output_query',
            nn.initializers.normal(c.prior_initial_scale, dtype),
            (1, patches.shape[-1])
        )
        outputs_q = io_processors.InputsMultiplexer(c.prior_initial_scale)(
            patches, low_dim
        )
        outputs_val = self.perceiver(inputs_q, outputs_q)
        patches, low_dim = io_processors.InputsMultiplexer.inverse(
            outputs_val, shapes=[patches_shape, ()]
        )
        voxels = self.voxels_proc.decode(patches, skip_connections)
        chex.assert_type([inputs_q, outputs_q, voxels], dtype)
        return self.action_decoder(voxels, low_dim)


# TODO: check if dtype is do something useful.
def _dtype_fromstr(dtype_str: str) -> types.DType:
    valid_dtypes = dict(bf16=jnp.bfloat16, f32=jnp.float32)
    return valid_dtypes[dtype_str]
