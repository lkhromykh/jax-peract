import chex
import jax.numpy as jnp


def _enc1d(dim_size: int, num_freqs: int, nyquist_freq: float):
    ls = jnp.linspace(-1, 1, dim_size)
    fs = jnp.linspace(1., nyquist_freq / 2., num_freqs)
    enc = jnp.pi * jnp.outer(ls, fs)
    ls = jnp.expand_dims(ls, -1)
    return jnp.concatenate([ls, jnp.cos(enc), jnp.sin(enc)], -1)


def _concat_emb(carry, y):
    """
    Args:
        carry.shape = (d1, d2, ..., dn, kx)
        y.shape = (dy, ky)
    Returns:
        res.shape = (d1, ..., dn, dy, kx + ky)
    """
    if not isinstance(carry, jnp.ndarray):
        # Carry init.
        return y
    x = carry
    dims = x.shape[:-1]
    x = jnp.expand_dims(x, -2)
    y = jnp.expand_dims(y, tuple(range(len(dims))))
    x = jnp.repeat(x, y.shape[-2], -2)
    repeats = dims + (1, 1)
    y = jnp.tile(y, repeats)
    return jnp.concatenate([x, y], -1)


def positional_encoding(x: chex.Array,
                        axis: tuple[int],
                        num_freqs: int,
                        nyquiste_freq: float
                        ) -> chex.Array:
    # Embedding dim scales as #axis(2num_freqs + 1)
    # Still need to assure that this produces desirable output
    # Batch dimensions are ignored since it produces the same encoding.
    enc = 1
    for size in axis:
        enc1d = _enc1d(x.shape[size], num_freqs, nyquiste_freq)
        enc = _concat_emb(enc, enc1d)
    return enc
