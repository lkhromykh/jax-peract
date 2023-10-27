import numpy as np


def fourier_features(shape: tuple[int, ...],
                     num_freqs: int,
                     nyquist_freqs: tuple[float, ...] | None = None
                     ) -> np.ndarray:
    ls = (np.linspace(-1., 1., ax) for ax in shape)
    pos = np.meshgrid(*ls, indexing='ij')
    pos = np.stack(pos, -1)
    nf = np.asarray(nyquist_freqs or shape)
    freqs = np.linspace(np.ones_like(nf), nf / 2., num_freqs)
    enc = np.pi * np.einsum('...d,Kd->...dK', pos, freqs)
    enc = enc.reshape(shape + (len(shape) * num_freqs,))
    return np.concatenate([pos, np.cos(enc), np.sin(enc)], -1)
