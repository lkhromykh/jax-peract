import numpy as np
import dm_env.specs

from rlbench.backend.scene import Scene
from rlbench.backend.observation import Observation
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import ActionMode as _ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning

from src import types_ as types
Array = types.Array


# TODO: robot should be sent voxels center
class ActionMode(_ActionMode):

    def __init__(self,
                 scene_bounds: tuple[Array, Array],
                 nbins: int
                 ) -> None:
        """Axis share number of bins just to simplify policy distribution.

        Action is described as [x, y, z, qw, qi, qj, qk, gripper_pos].
        """
        super().__init__(EndEffectorPoseViaPlanning(), Discrete())
        lb = np.concatenate([scene_bounds[0], -1, -1, -1, -1, 0])
        ub = np.concatenate([scene_bounds[1], 1, 1, 1, 1, 1])
        self._action_bounds = lb, ub
        self.nbins = nbins
        self._actgrid = np.linspace(lb, ub, nbins)

    def action(self, scene: Scene, action: types.Action) -> None:
        action = np.take_along_axis(self._actgrid, action[np.newaxis], 0)
        pos, quat, grip = np.split(action.squeeze(0), [3, 7])
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
        lb, ub = self._action_bounds
        return dm_env.specs.BoundedArray(lb.shape, np.int32, 0, self.nbins,
                                         name='discrete multicategorical')

    def from_observation(self, obs: Observation) -> types.Action:
        scene_bounds, (qlb, qub), (glb, gub) = map(
            lambda x: np.split(x, [3, 7]),
            self._action_bounds
        )
        pos = np.argmax(obs.gripper_pose > scene_bounds, 0)
        quat = mat_to_quat(obs.gripper_matrix)
        quat = np.floor_divide(quat - qlb, qub - qlb)
        grip = np.floor_divide(obs.gripper_pose - glb, gub - glb)
        return np.concatenate([pos, quat, grip], -1)
