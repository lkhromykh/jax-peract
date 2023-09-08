import collections.abc

import numpy as np
import dm_env.specs

Array = np.ndarray
RNG = np.random.RandomState

Observation = collections.abc.Mapping[str, Array]
Action = collections.abc.Mapping[str, Array]
ObservationSpec = collections.abc.Mapping[str, dm_env.specs.Array]
ActionSpec = collections.abc.Mapping[str, dm_env.specs.BoundedArray | dm_env.specs.DiscreteArray]

Layers = collections.abc.Sequence[int]
Metrics = collections.abc.MutableMapping[str, float]
