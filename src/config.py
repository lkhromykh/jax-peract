import dataclasses


@dataclasses.dataclass
class Config:

    num_bands: int = 5
    mixup_lambda: float = 0.

    # Perceiver
    latent_dim: int = 32
    latent_channels: int = 64
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
    weight_decay: float = 1e-5
    training_steps: int = 10 ** 6
    eval_every: int = 10000
    jit: bool = True
    precision: str = 'p=32,c=32,o=32'

    # Environment
    scene_lower_bound: list[float] = (-0.25, -0.5, 0.75)
    scene_upper_bound: list[float] = (0.75, 0.5, 1.75)
    time_limit: int = 10

    seed: int = 1
    logdir: str = 'logdir/'
