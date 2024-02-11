import collections.abc
from typing import Any, Callable, NamedTuple, TypeAlias, TypedDict

import jax
from dm_env import specs

Array: TypeAlias = jax.Array
DType: TypeAlias = Any
Action: TypeAlias = jax.Array
ActionSpec: TypeAlias = list[specs.DiscreteArray]
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
StepFn = Callable[['TrainState', Trajectory], tuple['TrainState', Metrics]]
