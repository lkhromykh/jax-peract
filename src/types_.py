import collections.abc
from typing import Any, TypedDict

import jax
import dm_env.specs

Array = RNG = jax.Array

Layers = collections.abc.Sequence[int]
Metrics = collections.abc.MutableMapping[str, float]

Action = Array
ActionSpec = list[dm_env.specs.DiscreteArray]
ObservationSpec = collections.abc.Mapping[str, dm_env.specs.Array]


class Observation(TypedDict):
    voxels: Array
    low_dim: Array
    task: Array


class Trajectory(TypedDict, total=False):
    observations: Any
    actions: Any
