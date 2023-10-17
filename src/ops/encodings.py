import numpy as np

Array = np.ndarray


def positional_encoding(x: Array,
                        axes: tuple[int, ...],
                        num_freqs: int,
                        nyquist_freqs: tuple[float, ...] | None = None
                        ) -> Array:
    shape = tuple(x.shape[ax] for ax in axes)
    ls = (np.linspace(-1., 1., ax) for ax in shape)
    pos = np.meshgrid(*ls, indexing='ij')
    pos = np.stack(pos, -1)
    nf = np.asarray(nyquist_freqs or shape)
    freqs = np.linspace(np.ones_like(nf), nf / 2., num_freqs)
    enc = np.pi * np.einsum('...d,Kd->...dK', pos, freqs)
    enc = enc.reshape(shape + (len(axes) * num_freqs,))
    return np.concatenate([pos, np.cos(enc), np.sin(enc)], -1)


def multimodal_encoding(obs: dict[str, Array],
                        ) -> dict[str, Array]:
    n_modalities = len(obs)

    def one_hot(idx):
        x = np.zeros(n_modalities, np.int32)
        x[idx] = 1
        return x
    enc = {}
    for i, (key, value) in enumerate(sorted(obs.items())):
        shape = value.shape[:-1] + (1,)
        enc[key] = np.tile(one_hot(i), shape)
    return enc
