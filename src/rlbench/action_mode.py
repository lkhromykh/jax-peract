import numpy as np
import dm_env.specs

from rlbench.backend.scene import Scene
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import ActionMode as _ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning

from src import types_ as types
Array = types.Array

DOWN_QUAT = np.array([0., 0.70710678, 0.70710678, 0.])


# TODO: robot should be sent voxels center
class ActionMode(_ActionMode):

    def __init__(self,
                 action_bounds: tuple[Array, Array],
                 nbins: int
                 ) -> None:
        """Axis share number of bins just to simplify policy distribution."""
        super().__init__(EndEffectorPoseViaPlanning(), Discrete())
        lb, ub = self._action_bounds = action_bounds
        self.nbins = nbins
        self._actgrid = np.linspace(lb, ub, nbins)

    def action(self, scene: Scene, action: Array) -> None:
        action = np.take_along_axis(self._actgrid, action[np.newaxis], 0)
        arm, grip = np.split(action.squeeze(0), [3])
        arm = np.concatenate([arm, DOWN_QUAT], -1)
        self.arm_action_mode.action(scene, arm)
        self.gripper_action_mode.action(scene, grip)

    def action_shape(self, scene: Scene) -> tuple[int, ...]:
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

    def action_bounds(self) -> tuple[Array, Array]:
        return self._action_bounds

    def action_spec(self) -> dm_env.specs.BoundedArray:
        lb, ub = self._action_bounds
        return dm_env.specs.BoundedArray(lb.shape, np.int32, 0, self.nbins,
                                         name='discrete multicategorical')
