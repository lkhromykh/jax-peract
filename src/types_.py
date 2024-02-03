import collections.abc
from typing import Any, NamedTuple, TypeAlias, TypedDict

from dm_env import specs
import jax

Array: TypeAlias = jax.Array
DType: TypeAlias = Any
RNG: TypeAlias = jax.Array
Action: TypeAlias = jax.Array
ActionSpec: TypeAlias = tuple[specs.DiscreteArray]
EnvSpecs = tuple['State[specs.Array]', ActionSpec]


class State(NamedTuple):
    """Extracted data required to make an informed action."""
    voxels: Array
    low_dim: Array
    goal: Array


class Trajectory(TypedDict, total=False):
    observations: Any
    actions: Any
    rewards: Any
    discounts: Any


SceneBounds: TypeAlias = tuple[float, float, float, float, float, float]
Layers = collections.abc.Sequence[int]
Metrics = collections.abc.MutableMapping[str, float]
