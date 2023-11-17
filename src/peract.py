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

    cfg: Config
    observation_spec: types.ObservationSpec
    action_spec: types.ActionSpec

    def setup(self) -> None:
        c = self.cfg
        dtype = _dtype_fromstr(c.compute_dtype)
        self.voxels_proc = nets.VoxelsProcessor(
            features=c.conv_stem_features,
            kernels=c.conv_stem_kernels,
            strides=c.conv_stem_strides,
            dtype=dtype,
            use_skip_connections=True,
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
            use_query_residual=c.use_query_residual,
            prior_initial_scale=c.prior_initial_scale,
            dtype=dtype,
            kernel_init=nn.initializers.lecun_normal(),
            use_layernorm=c.use_layernorm
        )
        self.action_decoder = nets.ActionDecoder(
            action_spec=self.action_spec,
            dtype=dtype
        )

    @nn.compact
    def __call__(self, obs: types.Observation) -> tfd.Distribution:
        chex.assert_rank([obs.voxels, obs.low_dim, obs.task],
                         [4, 1, 1])
        chex.assert_type([obs.voxels, obs.low_dim, obs.task],
                         [jnp.uint8, jnp.float16, jnp.int32])
        c = self.cfg
        dtype = _dtype_fromstr(c.compute_dtype)
        voxels = obs.voxels.astype(dtype) / 128. - 1
        voxels, skip_connections = self.voxels_proc.encode(voxels)
        pos3d_enc = utils.fourier_features(voxels.shape[:3], c.ff_num_bands)
        voxels = jnp.concatenate([voxels, pos3d_enc], -1)
        voxels = voxels.reshape(-1, voxels.shape[-1])
        low_dim = obs.low_dim.reshape(1, -1)
        task = obs.task.reshape(1, -1)
        voxels, low_dim, task = map(lambda x: x.astype(dtype),
                                    (voxels, low_dim, task))

        inputs_q = nets.InputsMultiplexer(c.prior_initial_scale)(
            voxels, low_dim, task
        )
        outputs_q = nets.InputsMultiplexer(c.prior_initial_scale)(
            voxels, low_dim
        )
        outputs_val = self.perceiver(inputs_q, outputs_q)
        voxels, low_dim = nets.InputsMultiplexer.inverse(
            outputs_val,
            skip_connections[-1].shape[:-1],
            ()
        )
        voxels = self.voxels_proc.decode(voxels, skip_connections)
        return self.action_decoder(voxels, low_dim)


def _dtype_fromstr(dtype_str: str) -> types.DType:
    valid_dtypes = dict(bf16=jnp.bfloat16, f32=jnp.float32)
    return valid_dtypes[dtype_str]
