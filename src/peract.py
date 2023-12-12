import jax.numpy as jnp
import flax.linen as nn
import chex
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from src import utils
import src.types_ as types
from src.config import Config
from src import networks as nets


class PerAct(nn.Module):

    config: Config
    observation_spec: types.ObservationSpec
    action_spec: types.ActionSpec

    def setup(self) -> None:
        c = self.config
        dtype = _dtype_fromstr(c.compute_dtype)
        self.voxels_proc = nets.VoxelsProcessor(
            features=c.conv_stem_features,
            kernels=c.conv_stem_kernels,
            strides=c.conv_stem_strides,
            dtype=dtype,
            use_skip_connections=c.conv_stem_use_skip_connections,
        )
        self.perceiver = nets.PerceiverIO(
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
        self.action_decoder = nets.ActionDecoder(
            action_spec=self.action_spec,
            mlp_layers=c.act_decoder_mlp_layers,
            conv_kernel=c.act_decoder_conv_kernel,
            dtype=dtype
        )

    @nn.compact
    def __call__(self, obs: types.Observation) -> tfd.Distribution:
        chex.assert_rank([obs.voxels, obs.low_dim, obs.task], [4, 1, 2])
        chex.assert_type([obs.voxels, obs.low_dim], [jnp.uint8, float])
        c = self.config
        dtype = _dtype_fromstr(c.compute_dtype)
        voxels, low_dim, task = map(lambda x: x.astype(dtype), obs)
        voxels = voxels / 128. - 1
        voxels, skip_connections = self.voxels_proc.encode(voxels)
        *vgrid_size, channels = voxels.shape
        vgrid_size = tuple(vgrid_size)
        pos3d_enc = utils.fourier_features(vgrid_size, c.ff_num_bands)
        if c.use_trainable_pos_encoding:  # Hide 3D structure of the voxels.
            pos3d_enc = self.param(
                'input_pos3d_enc',
                nn.initializers.normal(c.prior_initial_scale, voxels.dtype),
                pos3d_enc.shape  # Make further capacity equivalent.
            )
        voxels = jnp.concatenate([voxels, pos3d_enc], -1)
        voxels = voxels.reshape(-1, voxels.shape[-1])
        low_dim = nn.Dense(channels, dtype=dtype)(low_dim).reshape(1, -1)
        task = nn.Dense(channels, dtype=dtype)(task)
        pos1d_enc = utils.fourier_features(task.shape[:1], c.ff_num_bands)
        task = jnp.concatenate([task, pos1d_enc], -1)

        inputs_q = nets.InputsMultiplexer(c.prior_initial_scale)(
            voxels, low_dim, task
        )
        if c.use_trainable_pos_encoding:
            voxels = self.param(
                'output_pos3d_enc',
                nn.initializers.normal(c.prior_initial_scale, voxels.dtype),
                voxels.shape
            )
        low_dim = self.param(
            'low_dim_output_q',
            nn.initializers.normal(c.prior_initial_scale, dtype),
            (1, voxels.shape[-1])
        )
        outputs_q = nets.InputsMultiplexer(c.prior_initial_scale)(
            voxels, low_dim
        )
        outputs_val = self.perceiver(inputs_q, outputs_q)
        voxels, low_dim = nets.InputsMultiplexer.inverse(
            outputs_val, shapes=[vgrid_size, ()]
        )
        voxels = self.voxels_proc.decode(voxels, skip_connections)
        chex.assert_type([inputs_q, outputs_q, voxels], dtype)
        return self.action_decoder(voxels, low_dim)


def _dtype_fromstr(dtype_str: str) -> types.DType:
    # f32 still lowered to bf16 via jax.lax.Precision.DEFAULT.
    valid_dtypes = dict(bf16=jnp.bfloat16, f32=jnp.float32)
    return valid_dtypes[dtype_str]
