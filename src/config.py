from typing import Literal
from rltools.config import Config as _Config

Layers = tuple[int, ...]


class Config(_Config):
    # Conv stem
    conv_stem_features: Layers = (32,)
    conv_stem_kernels: Layers = (4,)
    conv_stem_strides: Layers = (4,)
    # Perceiver
    latent_dim: int = 64
    latent_channels: int = 32
    num_blocks: int = 1
    num_self_attend_per_block: int = 6
    num_cross_attend_heads: int = 1
    num_self_attend_heads: int = 1
    cross_attend_widening_factor: float = 1.
    self_attend_widening_factor: float = 1.
    use_query_residual: bool = True
    use_layernorm: bool = True
    prior_initial_scale: float = 0.02
    ff_num_bands: int = 0
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_grad_norm: float = 10.
    weight_decay: float = 0.
    training_steps: int = 10 ** 6
    eval_every: int = 1000
    jit: bool = True
    compute_dtype: Literal['bf16', 'f32'] = 'bf16'
    # Environment
    scene_lower_bound: list[float] = (-0.3, -0.5, 0.6)
    scene_upper_bound: list[float] = (0.7, 0.5, 1.6)
    scene_bins: int = 32
    rot_bins: int = 7
    time_limit: int = 10
    num_demos: int = 50

    seed: int = 1
    logdir: str = 'logdir/'
