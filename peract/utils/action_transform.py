import numpy as np
from dm_env import specs

from peract.environment import gcenv
import peract.types_ as types

Discrete = np.ndarray
_eps = 1e-5


class DiscreteActionTransform:
    """Converts robot actions to a uniformly discretized action.

    Incoming action must meet the gcenv.ActionSpec format:
    [x, y, z, yaw, pitch, roll, grasp, terminate].
    """

    def __init__(self,
                 scene_bounds: gcenv.SceneBounds,
                 scene_bins: int,
                 rot_bins: int,
                 ) -> None:
        rot_lim = np.array([np.pi, np.pi / 2, np.pi])
        lb = np.r_[scene_bounds[:3], -rot_lim, 0, 0]
        ub = np.r_[scene_bounds[3:], rot_lim, 1, 1]
        nbins = 3 * [scene_bins] + 3 * [rot_bins] + [2, 2]
        self._act_specs = tuple(specs.DiscreteArray(n) for n in nbins)
        self._action_bounds = (lb, ub)
        self._range = ub - lb
        self._nbins = np.int32(nbins)

    def encode(self, action: gcenv.Action) -> Discrete:
        assert action.shape == (8,)
        lb, ub = self._action_bounds
        action = (action - lb) / self._range
        action = np.clip(action, a_min=_eps, a_max=1. - _eps)
        action = np.floor(self._nbins * action).astype(np.int32)
        self._assert_valid_action(action)
        return action

    def decode(self, action: Discrete) -> gcenv.Action:
        self._assert_valid_action(action)
        lb, _ = self._action_bounds
        action = lb + self._range * (action + 0.5) / self._nbins
        return action.astype(np.float32)

    def action_spec(self) -> types.ActionSpec:
        return self._act_specs

    def get_scene_center(self) -> np.ndarray:
        lb, ub = self._action_bounds
        return (lb[:3] + ub[:3]) / 2

    def _assert_valid_action(self, action: gcenv.Action) -> None:
        assert action.shape == np.shape(self._nbins) \
           and action.dtype == np.int32 \
           and np.all(action >= 0) \
           and np.all(action <= self._nbins - 1), \
               action
