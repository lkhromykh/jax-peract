import dataclasses
from rltools.config import Config as _Config

Layers = tuple[int, ...]


@dataclasses.dataclass
class Config(_Config):
    # Conv stem
    conv_stem_features: Layers = (64,)
    conv_stem_kernels: Layers = (4,)
    conv_stem_strides: Layers = (4,)
    conv_stem_use_skip_connections: bool = True
    # Perceiver
    latent_dim: int = 256
    latent_channels: int = 128
    num_blocks: int = 1
    num_self_attend_per_block: int = 6
    num_cross_attend_heads: int = 1
    num_self_attend_heads: int = 8
    cross_attend_widening_factor: float = 1.
    self_attend_widening_factor: float = 1.
    use_layernorm: bool = True
    use_decoder_query_residual: bool = False
    use_trainable_pos_encoding: bool = False
    prior_initial_scale: float = 0.02
    ff_num_bands: int = 16
    text_emb_len: int = -1  # 20 (max 77)
    # Action decoder
    act_decoder_mlp_layers: Layers = (64,)
    act_decoder_conv_kernel: int = 3
    # Training
    max_grad_norm: float = 1.
    warmup_steps: int = 3000
    peak_learning_rate: float = 1e-4
    training_steps: int = 100_000
    batch_size: int = 16
    weight_decay: float = 1e-4
    eval_every: int = 500
    jit: bool = True
    compute_dtype: str = 'bf16'
    max_shift: int = 4
    # Environment
    scene_bounds: tuple[float, ...] = (-0.3, -0.5, 0.6, 0.7, 0.5, 1.6)
    scene_bins: int = 40
    rot_bins: int = 11
    time_limit: int = 16
    num_demos: int = 50

    seed: int = 1
    launch_env: bool = True
    logdir: str = 'logdir/open_door_40'


peract_config = Config(
    conv_stem_features=(64, 64,),
    conv_stem_kernels=(1, 5),
    conv_stem_strides=(1, 5),
    conv_stem_use_skip_connections=True,
    latent_dim=2048,
    latent_channels=512,
    num_blocks=1,
    num_self_attend_per_block=6,
    num_cross_attend_heads=1,
    num_self_attend_heads=8,
    cross_attend_widening_factor=4.,
    self_attend_widening_factor=4.,
    use_trainable_pos_encoding=True,
    text_emb_len=77,
    act_decoder_mlp_layers=(256,),
    act_decoder_conv_kernel=3,
    training_steps=600_000,
    batch_size=16,
    warmup_steps=3000,
    weight_decay=1e-6,
    rot_bins=72,
    scene_bins=100,
)

