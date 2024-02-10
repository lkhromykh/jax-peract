import numpy as np
from dm_env import specs

from src.environment import gcenv
import src.types_ as types

Discrete = np.ndarray


class DiscreteActionTransform:
    """Converts robot actions to a uniformly discretized action.

    Encoded action has the form: [x, y, z, yaw, pitch, roll, grip, terminate].
    """

    def __init__(self,
                 scene_bounds: gcenv.SceneBounds,
                 scene_bins: int,
                 rot_bins: int,
                 grip_bins: int
                 ) -> None:
        rot_lim = np.array([np.pi, np.pi / 2, np.pi])
        lb = np.r_[scene_bounds[:3], -rot_lim, 0, 0]
        ub = np.r_[scene_bounds[3:], rot_lim, 1, 1]
        nbins = 3 * [scene_bins] + 3 * [rot_bins] + [grip_bins, 2]
        self._act_specs = [specs.DiscreteArray(n) for n in nbins]
        self._action_bounds = (lb, ub)
        self._range = ub - lb
        self._nbins = np.int32(nbins) - 1  # indexing starts from 0.

    def encode(self, action: gcenv.Action) -> Discrete:
        assert action.shape == (8,)
        lb, ub = self._action_bounds
        action = np.clip(action, a_min=lb, a_max=ub)
        action = (action - lb) / self._range
        action = np.round(self._nbins * action).astype(np.int32)
        self._assert_valid_action(action)
        return action

    def decode(self, action: Discrete) -> gcenv.Action:
        self._assert_valid_action(action)
        lb, _ = self._action_bounds
        action = lb + self._range * action / self._nbins
        return action

    def action_spec(self) -> types.ActionSpec:
        return self._act_specs.copy()

    def _assert_valid_action(self, action: gcenv.Action) -> None:
        assert action.shape == np.shape(self._nbins) \
           and action.dtype == np.int32 \
           and np.all(action >= 0) \
           and np.all(action <= self._nbins), \
               action
