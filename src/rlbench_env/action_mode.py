from collections import namedtuple

import numpy as np
import dm_env.specs

from rlbench.backend.scene import Scene
from rlbench.backend.observation import Observation
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import ActionMode as _ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning

from src import types_ as types

Array = types.Array


class ActionMode(_ActionMode):

    Discretization = namedtuple('Discretization', ('pos', 'rot', 'grip'))

    def __init__(self,
                 scene_bounds: tuple[Array, Array],
                 discretization: Discretization
                 ) -> None:
        """Axis share number of bins just to simplify policy distribution.

        Action is described as [x, y, z, qw, qi, qj, qk, gripper_pos].
            and discretized to #[  pos,        rot,          grip    ] bins.
        """
        super().__init__(EndEffectorPoseViaPlanning(), Discrete())
        self.scene_bounds = scene_bounds
        self.discretization = d = discretization
        lb = np.concatenate([scene_bounds[0], [-1, -1, -1, -1, 0]])
        ub = np.concatenate([scene_bounds[1], [1, 1, 1, 1, 1]])
        self._nbins = np.int32(3 * [d.pos] + 4 * [d.rot] + [d.grip]) - 1
        self._action_bounds = lb, ub
        self._range = ub - lb

    def action(self, scene: Scene, action: types.Action) -> None:
        lb, ub = self._action_bounds
        action = np.asarray(action, np.int32)
        action = lb + self._range * action / self._nbins
        pos, quat, grip = np.split(action, [3, 7])
        quat /= np.linalg.norm(quat)
        arm = np.concatenate([pos, quat], -1)
        self.arm_action_mode.action(scene, arm)
        self.gripper_action_mode.action(scene, grip)

    def action_shape(self, scene: Scene) -> tuple[int, ...]:
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

    def action_bounds(self) -> tuple[Array, Array]:
        return self._action_bounds

    def action_spec(self) -> types.ActionSpec:
        def spec(num_values): return dm_env.specs.DiscreteArray(num_values)
        return [spec(num + 1) for num in self._nbins]  # thus factorized.

    def from_observation(self, obs: Observation) -> types.Action:
        # This is possible
        lb, ub = self._action_bounds
        action = np.concatenate([obs.gripper_pose, [1. - obs.gripper_open]])
        action = np.clip(action, a_min=lb, a_max=ub)
        action = (action - lb) / self._range
        action = np.round(self._nbins * action).astype(np.int32)
        assert np.all(action >= 0) and np.all(action <= self._nbins), action
        return action
