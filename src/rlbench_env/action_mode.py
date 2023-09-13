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

    def __init__(self,
                 scene_bounds: tuple[Array, Array],
                 nbins: int
                 ) -> None:
        """Axis share number of bins just to simplify policy distribution.

        Action is described as [x, y, z, qw, qi, qj, qk, gripper_pos].
        Each action value lies in range(nbins).
        """
        super().__init__(EndEffectorPoseViaPlanning(), Discrete())
        lb = np.concatenate([scene_bounds[0], [-1, -1, -1, -1, 0]])
        ub = np.concatenate([scene_bounds[1], [1, 1, 1, 1, 1]])
        self._action_bounds = lb, ub
        self.nbins = nbins
        self._actgrid = np.linspace(lb, ub, nbins).T

    def action(self, scene: Scene, action: types.Action) -> None:
        action = np.take_along_axis(self._actgrid, np.expand_dims(action, 1), 1)
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
        lb, _ = self._action_bounds
        return dm_env.specs.BoundedArray(shape=lb.shape, dtype=np.int32,
                                         minimum=0, maximum=self.nbins,
                                         name='multicategorical')

    def from_observation(self, obs: Observation) -> types.Action:
        (plb, qlb, glb), (pub, qub, gub) = map(
            lambda x: np.split(x, [3, 7]),
            self._action_bounds
        )
        pos, quat = np.split(obs.gripper_pose, [3])

        def discretize(x, a_min, a_max):
            x = np.clip(x, a_min, a_max - 1e-7)
            return np.floor_divide((x - a_min) * self.nbins,
                                   (a_max - a_min))
        pos = discretize(pos, plb, pub)
        quat = discretize(quat, qlb, qub)
        grip = discretize(obs.gripper_open, glb, gub)
        action = np.concatenate([pos, quat, grip], -1).astype(np.int32)
        assert np.all(action >= 0) and np.all(action < self.nbins), action
        return action
