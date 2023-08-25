import dataclasses


@dataclasses.dataclass
class Config:

    img_size: tuple[int, int] | None = None
    num_freqs: int = 64
    nyquist_freq: float = 32

    # Perceiver
    latent_dim: int = 256
    latent_channels: int = 128  # should be divisible by lt_num_heads
    feedforward_dim: int = 256
    num_blocks: int = 2
    num_heads: int = 4

    # Latent transformer
    lt_num_blocks: int = 6
    lt_feedforward_dim: int = 256
    lt_num_heads: int = 8

    # Training
    batch_size: int = 64
    learning_rate: float = 6e-4
    max_grad_norm: float = 10.
    weight_decay: float = 1e-4
    training_steps: int = 10 ** 5
    eval_every: int = 1000
    jit: bool = True

    logdir: str = 'logdir'
    seed: int = 0
    num_classes: int = 10
