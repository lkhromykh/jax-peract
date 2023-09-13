import collections.abc
from typing import Any, TypedDict

import numpy as np
import dm_env.specs

Array = np.ndarray
RNG = np.random.RandomState

Layers = collections.abc.Sequence[int]
Metrics = collections.abc.MutableMapping[str, float]

Action = Array
ActionSpec = dm_env.specs.DiscreteArray
ObservationSpec = collections.abc.Mapping[str, dm_env.specs.Array]


# Latter classes are not used directly since they are not jax.PyTree's.
class Observation(TypedDict, total=False):
    voxels: Array
    low_dim: Array
    task: Array


class Trajectory(TypedDict):
    observations: Any
    actions: Any
