import chex
import jax.numpy as jnp


def _enc1d(dim_size: int, num_freqs: int, nyquist_freq: float):
    ls = jnp.linspace(-1, 1, dim_size)
    fs = jnp.linspace(1., nyquist_freq / 2., num_freqs)
    enc = jnp.pi * jnp.outer(ls, fs)
    ls = jnp.expand_dims(ls, -1)
    return jnp.concatenate([ls, jnp.cos(enc), jnp.sin(enc)], -1)


def _concat_emb(carry, x):
    """
    Args:
        carry_n.shape = (d1, d2, ..., dn, kn)
        x.shape = (dx, kx)
    Returns:
        carry_{n+1}.shape = (d1, ..., dn, dx, kn + kx)
    """
    if not isinstance(carry, jnp.ndarray):
        # Carry init.
        return x
    c = carry
    dims = c.shape[:-1]
    c = jnp.expand_dims(c, -2)
    x = jnp.expand_dims(x, tuple(range(len(dims))))
    c = jnp.repeat(c, x.shape[-2], -2)
    repeats = dims + (1, 1)
    x = jnp.tile(x, repeats)
    return jnp.concatenate([c, x], -1)


def positional_encoding(x: chex.Array,
                        axis: tuple[int],
                        num_freqs: int,
                        nyquiste_freq: float
                        ) -> chex.Array:
    # Embedding dim scales as #axis(2num_freqs + 1)
    # Batch dimensions are ignored since it produces the same encoding.
    enc = 1
    for size in axis:
        enc1d = _enc1d(x.shape[size], num_freqs, nyquiste_freq)
        enc = _concat_emb(enc, enc1d)
    return enc
