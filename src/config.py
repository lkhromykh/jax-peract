import dataclasses
from rltools.config import Config as _Config

Layers = tuple[int, ...]


@dataclasses.dataclass
class Config(_Config):
    # Conv stem
    conv_stem_features: Layers = (32,)
    conv_stem_kernels: Layers = (4,)
    conv_stem_strides: Layers = (4,)
    # Perceiver
    latent_dim: int = 32
    latent_channels: int = 32
    num_blocks: int = 1
    num_self_attend_per_block: int = 6
    num_cross_attend_heads: int = 1
    num_self_attend_heads: int = 1
    cross_attend_widening_factor: float = 1.
    self_attend_widening_factor: float = 1.
    use_query_residual: bool = True
    use_layernorm: bool = True
    use_trainable_pos_encoding: bool = False
    prior_initial_scale: float = 0.02
    ff_num_bands: int = 10
    text_emb_len: int = -1  # 20 (max 77)
    # Training
    max_grad_norm: float = 10.
    warmup_steps: int = 6000
    peak_learning_rate: float = 5e-4
    training_steps: int = 600_000
    batch_size: int = 16
    weight_decay: float = 1e-4
    eval_every: int = 500
    jit: bool = True
    compute_dtype: str = 'f32'
    max_shift: int = 4
    # Environment
    scene_bounds: tuple[float, ...] = (-0.3, -0.5, 0.7, 0.7, 0.5, 1.7)
    scene_bins: int = 32
    rot_bins: int = 7
    time_limit: int = 10
    num_demos: int = 50

    seed: int = 1
    launch_env: bool = True
    logdir: str = 'logdir/pick_and_lift'


peract_config = Config(
    conv_stem_features=(64,),
    conv_stem_kernels=(5,),
    conv_stem_strides=(5,),
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
    training_steps=600_000,
    batch_size=16,
    warmup_steps=3000,
    weight_decay=1e-6,
    rot_bins=72,
    scene_bins=100,
)

