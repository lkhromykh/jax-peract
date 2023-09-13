from collections.abc import Iterable

import numpy as np

Array = np.ndarray


def positional_encoding(x: Array,
                        axes: tuple[int, ...],
                        num_freqs: int,
                        nyquist_freqs: float | tuple[float, ...]
                        ) -> Array:
    shape = tuple(x.shape[ax] for ax in axes)
    ls = (np.linspace(-1., 1., ax, endpoint=True) for ax in shape)
    pos = np.meshgrid(*ls, indexing='ij')
    pos = np.stack(pos, -1)
    if isinstance(nf := nyquist_freqs, Iterable):
        nf = np.asarray(nf)
    else:
        nf = nf * np.ones(len(axes))
    freqs = np.linspace(np.ones_like(nf), nf / 2., num_freqs, endpoint=True)
    enc = np.pi * np.einsum('...d,Kd->...dK', pos, freqs)
    enc = enc.reshape(shape + (len(axes) * num_freqs,))
    return np.concatenate([pos, np.cos(enc), np.sin(enc)], -1)


def multimodal_encoding(obs: dict[str, Array]) -> dict[str, Array]:
    n_modalities = len(obs)

    def one_hot(idx):
        x = np.zeros(n_modalities, np.int32)
        x[idx] = 1
        return x
    enc = {}
    for i, (key, value) in enumerate(sorted(obs.items())):
        shape = value.shape[:-1] + (1,)
        modality = np.tile(one_hot(i), shape)
        enc[key] = modality
    return enc
