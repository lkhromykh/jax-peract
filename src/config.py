import dataclasses


@dataclasses.dataclass
class Config:

    img_size: tuple[int, int] = (32, 32)
    num_freqs: int = 32
    nyquist_freq: float = 32

    # Perceiver
    latent_dim: int = 32
    latent_channels: int = 32
    feedforward_dim: int = 32
    num_blocks: int = 1
    num_heads: int = 4

    # Latent transformer
    lt_num_blocks: int = 1
    lt_feedforward_dim: int = 32
    lt_num_heads: int = 4

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_grad_norm: float = 10.
    weight_decay: float = 1e-2
    training_steps: int = 10 ** 5
    jit: bool = True

    logdir: str = 'logdir'
    seed: int = 0
    num_classes: int = 10
