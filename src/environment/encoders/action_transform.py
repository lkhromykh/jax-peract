from typing import Literal

import numpy as np
from dm_env.specs import DiscreteArray
from scipy.spatial.transform import Rotation as R

from src.environment import base


Continuous = Discrete = base.Array
RotRepr = Literal['euler', 'quat', 'rotvec']
SceneBounds = tuple[float, float, float, float, float, float]


class DiscreteActionTransform:
    """Converts robot actions to a uniformly discretized action.

    Encoded action has the form: [*xyz, *euler_rot, grip, terminate] (size=8).
    """

    def __init__(self,
                 scene_bounds: SceneBounds,
                 scene_bins: int,
                 rot_bins: int,
                 rot_repr: RotRepr
                 ) -> None:
        rot_limits = np.array([np.pi, np.pi / 2, np.pi])
        lb = np.r_[scene_bounds[:3], -rot_limits, 0, 0]
        ub = np.r_[scene_bounds[3:],  rot_limits, 1, 1]
        nbins = 3 * [scene_bins] + 3 * [rot_bins] + [2, 2]
        self.rot_repr = rot_repr
        self._specs = [Discrete(n) for n in nbins]
        self._action_bounds = lb, ub
        self._range = ub - lb
        self._nbins = np.int32(nbins) - 1  # indexing starts from 0.

    def encode(self, action: Continuous) -> Discrete:
        lb, ub = self._action_bounds
        pos, rot, grip = np.split(action, [3, -1])
        match self.rot_repr:
            case 'euler': rot = R.from_euler(rot, 'xyz')
            case 'quat': rot = R.from_quat(rot)
            case 'rotvec': rot = R.from_rotvec(rot, degrees=False)
            case _: raise ValueError(self.rot_repr)
        euler = rot.as_euler('xyz')
        action = np.concatenate([pos, euler, grip])
        action = np.clip(action, a_min=lb, a_max=ub)
        action = (action - lb) / self._range
        action = np.round(self._nbins * action).astype(np.int32)
        self._assert_valid_action(action)
        return action

    def decode(self, action: Discrete) -> Continuous:
        self._assert_valid_action(action)
        lb, _ = self._action_bounds
        action = lb + self._range * action / self._nbins
        pos, euler, grip = np.split(action, [3, 6])
        rot = R.from_euler('xyz', euler)
        match self.rot_repr:
            case 'euler': rot = rot.as_euler(rot, 'xyz')
            case 'quat': rot = rot.as_quat(canonical=True)
            case 'rotvec': rot = rot.as_rotvec(degrees=False)
            case _: raise ValueError(self.rot_repr)
        return np.concatenate([pos, rot, grip])

    def action_spec(self) -> list[DiscreteArray]:
        return self._specs.copy()

    def _assert_valid_action(self, action: base.Action) -> None:
        assert action.shape == np.shape(self._nbins) \
           and action.dtype == np.int32 \
           and np.all(action >= 0) \
           and np.all(action <= self._nbins), \
               action
