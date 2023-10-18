import numpy as np
import dm_env.specs

from rlbench.backend.scene import Scene
from rlbench.backend.observation import Observation
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning

from src import types_ as types

Array = types.Array


class DiscreteActionMode(ActionMode):

    SCENE_BINS = 20
    ROT_BINS = 5
    GRIP_BINS = 2

    def __init__(self,
                 scene_bounds: tuple[Array, Array],
                 ) -> None:
        """Axis share number of bins just to simplify policy distribution.

        Action is described as [x, y, z, qw, qi, qj, qk, gripper_pos].
            and discretized to #[  pos,        rot,          grip    ] bins.
        """
        super().__init__(EndEffectorPoseViaPlanning(), Discrete())
        self.scene_bounds = scene_bounds
        lb = np.concatenate([scene_bounds[0], [-1, -1, -1, -1, 0]])
        ub = np.concatenate([scene_bounds[1], [1, 1, 1, 1, 1]])
        nbins = 3 * [self.SCENE_BINS] + 4 * [self.ROT_BINS] + [self.GRIP_BINS]
        self._action_bounds = lb, ub
        self._range = ub - lb
        self._nbins = np.int32(nbins) - 1

    def action(self, scene: Scene, action: types.Action) -> None:
        self._assert_valid_action(action)
        lb, ub = self._action_bounds
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
        Spec = dm_env.specs.DiscreteArray
        scene_spec = Spec(self.SCENE_BINS ** 3)
        rot_specs = 4 * [Spec(self.ROT_BINS)]
        grip_spec = Spec(self.GRIP_BINS)
        return [scene_spec] + rot_specs + [grip_spec]

    def from_observation(self, obs: Observation) -> types.Action:
        lb, ub = self._action_bounds
        # maybe handle quaternion sign: if q[-1] < 0 then q <- -q
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        action = np.clip(action, a_min=lb, a_max=ub)
        action = (action - lb) / self._range
        action = np.round(self._nbins * action).astype(np.int32)
        self._assert_valid_action(action)
        return action

    def _assert_valid_action(self, action: Array) -> None:
        assert action.shape == (8,) \
           and action.dtype == np.int32 \
           and np.all(action >= 0) \
           and np.all(action <= self._nbins), \
               action
