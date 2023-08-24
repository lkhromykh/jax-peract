import numpy as np

Array = np.ndarray


def _enc1d(dim_size: int, num_freqs: int, nyquist_freq: float):
    ls = np.linspace(-1, 1, dim_size)
    fs = np.linspace(1., nyquist_freq / 2., num_freqs, endpoint=True)
    enc = np.pi * np.outer(ls, fs)
    ls = np.expand_dims(ls, -1)
    return np.concatenate([ls, np.cos(enc), np.sin(enc)], -1)


def _concat_emb(carry: Array, x: Array) -> Array:
    """
    Args:
        carry_n.shape = (d1, d2, ..., dn, kn)
        x.shape = (dx, kx)
    Returns:
        carry_{n+1}.shape = (d1, ..., dn, dx, kn + kx)
    """
    if not carry.size:
        return x  # Carry init.
    c = carry
    dims = c.shape[:-1]
    c = np.expand_dims(c, -2)
    x = np.expand_dims(x, tuple(range(len(dims))))
    c = np.repeat(c, x.shape[-2], -2)
    repeats = dims + (1, 1)
    x = np.tile(x, repeats)
    return np.concatenate([c, x], -1)


def positional_encoding(x: Array,
                        axis: tuple[int, ...],
                        num_freqs: int,
                        nyquiste_freq: float
                        ) -> Array:
    # Using numpy everywhere since
    #   this can be considered as a constant in a training loop.
    # Embedding dim scales as #axis(2num_freqs + 1)
    # Batch dimensions are ignored since it produces the same encoding.
    enc = np.array([])
    for size in axis:
        enc1d = _enc1d(x.shape[size], num_freqs, nyquiste_freq)
        enc = _concat_emb(enc, enc1d)
    return enc
