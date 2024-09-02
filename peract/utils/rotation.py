"""[1812.07035] On the Continuity of Rotation Representations in Neural Networks."""
import numpy as np
from scipy.spatial.transform import Rotation as _Rotation

_6DContinuous = np.ndarray


class Rotation(_Rotation):

    def as_continuous6d(self) -> _6DContinuous:
        assert self.single, 'Batching is not supported.'
        return self.as_matrix().T.flatten()[:6]

    @classmethod
    def from_continuous6d(cls, cont6d: _6DContinuous) -> 'Rotation':
        assert cont6d.shape == (6,), 'Batching is not supported.'
        a1, a2 = np.split(cont6d, 2)
        b1 = _safe_norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = _safe_norm(b2)
        b3 = np.cross(b1, b2)
        rmat = np.stack([b1, b2, b3]).T
        return Rotation.from_matrix(rmat)


def _safe_norm(x):
    return x / np.linalg.norm(x)
