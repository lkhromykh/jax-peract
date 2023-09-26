import dataclasses


@dataclasses.dataclass
class Config:

    num_bands: int = 5
    mixup_lambda: float = 0.

    # Perceiver
    latent_dim: int = 32
    latent_channels: int = 32
    cross_attend_blocks: int = 1
    self_attend_blocks: int = 4
    cross_attend_heads: int = 1
    self_attend_heads: int = 1
    cross_attend_widening_factor: int = 1
    self_attend_widening_factor: int = 1
    prior_initial_scale: float = 0.02

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_grad_norm: float = 10.
    weight_decay: float = 1e-4
    training_steps: int = 10 ** 6
    eval_every: int = 10000
    jit: bool = True

    # Environment
    scene_nbins: int = 10
    rot_nbins: int = 7
    grip_nbins: int = 2
    scene_lower_bound: list[float] = (-0.3, -0.5, 0.6)
    scene_upper_bound: list[float] = (0.7, 0.5, 1.6)

    seed: int = 1
    time_limit: int = float('inf')
    precision: str = 'p=32,c=32,o=32'
    logdir: str = 'logdir'
