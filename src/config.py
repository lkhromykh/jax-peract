import dataclasses


@dataclasses.dataclass
class Config:
    img_size: tuple[int, int] = (32, 32)
    num_freqs: int = 16
    nyquist_freq: float = 16

    latent_dim: int = 64
    dim_feedforward: int = 64
    num_layers: int = 2
    num_heads: int = 4
    activation: str = 'relu'

    batch_size: int = 32
    learning_rate: float = 1e-3
    max_grad_norm: float = 50.
    training_steps: int = 10 ** 4
    jit: bool = True

    logdir: str = 'logdir'
    seed: int = 0
