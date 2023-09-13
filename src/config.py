import dataclasses


@dataclasses.dataclass
class Config:

    num_freqs: int = 64
    nyquist_freq: float = 32
    mixup_lambda: float = 0.

    # Perceiver
    latent_dim: int = 32
    latent_channels: int = 32  # should be divisible by lt_num_heads
    feedforward_dim: int = 32
    num_blocks: int = 2
    num_heads: int = 4

    # Latent transformer
    lt_num_blocks: int = 6
    lt_feedforward_dim: int = 32
    lt_num_heads: int = 8

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-2
    max_grad_norm: float = 10.
    weight_decay: float = 1e-5
    training_steps: int = 10 ** 5
    eval_every: int = 1000
    jit: bool = True

    seed: int = 0
    logdir: str = 'logdir'
