from typing import Literal
from rltools.config import Config as _Config

Layers = tuple[int, ...]


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
    prior_initial_scale: float = 0.02
    ff_num_bands: int = 1
    text_emb_len: int = 20  # max 77
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.
    weight_decay: float = 0.01
    training_steps: int = 10 ** 6
    eval_every: int = 500
    jit: bool = True
    compute_dtype: Literal['bf16', 'f32'] = 'bf16'
    max_shift: int = 0
    ent_coef: float = 1e-3
    # Environment
    scene_bounds: tuple[float, ...] = (-0.3, -0.5, 0.6, 0.7, 0.5, 1.6)
    scene_bins: int = 32
    rot_bins: int = 7
    time_limit: int = 10
    num_demos: int = 20

    seed: int = 1
    launch_env: bool = True
    logdir: str = 'logdir/w_euler'
