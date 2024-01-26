import collections.abc
from typing import Any, NamedTuple, TypedDict

import jax

Array = RNG = jax.Array
DType = Any


class State(NamedTuple):
    """Processed data required to make an informed action."""
    voxels: Array
    low_dim: Array
    goal: dict[str, Array]


class Trajectory(TypedDict, total=False):
    observations: Any
    actions: Any
    rewards: Any
    discounts: Any


SceneBounds = tuple[float, float, float, float, float, float]
Layers = collections.abc.Sequence[int]
Metrics = collections.abc.MutableMapping[str, float]
