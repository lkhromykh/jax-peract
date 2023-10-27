import collections.abc
from typing import Any, NamedTuple, TypedDict

import jax
import dm_env.specs

Array = RNG = jax.Array

Layers = collections.abc.Sequence[int]
Metrics = collections.abc.MutableMapping[str, float]

Action = Array
ActionSpec = list[dm_env.specs.DiscreteArray]
ObservationSpec = tuple[Array]


class Observation(NamedTuple):
    voxels: Array
    low_dim: Array
    task: Array


class Trajectory(TypedDict, total=False):
    observations: Any
    actions: Any
    rewards: Any
    discounts: Any
