import collections.abc
from typing import TypedDict

import numpy as np
import dm_env.specs

Array = np.ndarray
RNG = np.random.RandomState

Layers = collections.abc.Sequence[int]
Metrics = collections.abc.MutableMapping[str, float]

Observation = collections.abc.Mapping[str, Array]
ObservationSpec = collections.abc.Mapping[str, dm_env.specs.Array]
Action = Array
ActionSpec = dm_env.specs.DiscreteArray


class Trajectory(TypedDict):
    observations: list[Observation]
    actions: list[Action]
