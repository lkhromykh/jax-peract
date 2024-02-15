import abc
import collections.abc
from typing import final, NamedTuple, TypeAlias

import tree
import numpy as np
import dm_env.specs

Array: TypeAlias = np.ndarray
Action: TypeAlias = np.ndarray
SceneBounds: TypeAlias = tuple[float, float, float, float, float, float]
Goal: TypeAlias = str | np.ndarray


# TODO: consider swapping to quat / rotvec to avoid Gimball lock.
# TODO: it may be handful to store extrinsic/intrinsic cam matrices.
class Observation(NamedTuple):
    """Complete action-centric environment state representation."""

    JOINTS_VEL_LOW_THRESHOLD = 0.1
    GRIPPER_OPEN_THRESHOLD = 0.05

    images: tuple[Array]  # N x (H_i, W_i, 3), N -- number of cam views.
    depth_maps: tuple[Array]  # N x (H, W)
    point_clouds: tuple[Array]  # N x (H, W, 3)
    joint_positions: Array
    joint_velocities: Array
    tcp_pose: Array  # [x, y, z, yaw, pitch, roll]
    gripper_pos: float  # [open=0, close=1]
    gripper_is_obj_detected: bool
    is_terminal: bool
    goal: Goal

    @property
    def gripper_is_open(self) -> bool:
        return self.gripper_pos < Observation.GRIPPER_OPEN_THRESHOLD

    @property
    def joints_velocity_is_low(self) -> bool:
        return np.allclose(self.joint_velocities, 0, atol=Observation.JOINTS_VEL_LOW_THRESHOLD)

    def infer_action(self) -> Action:
        """Extract action from an observation."""
        return np.r_[self.tcp_pose, 1. - self.gripper_is_open, self.is_terminal].astype(np.float32)

    def replace(self, **kwargs) -> 'Observation':
        return self._replace(**kwargs)


ActionSpec: TypeAlias = dm_env.specs.BoundedArray
ObservationSpec: TypeAlias = 'Observation[dm_env.specs.Array]'
Demo = collections.abc.Sequence[Observation]


class GoalConditionedEnv(dm_env.Environment):
    """Goal-conditioned with an ability to produce expert demonstrations."""

    def __init__(self,
                 scene_bounds: SceneBounds,
                 time_limit: int = float('inf')
                 ) -> None:
        self.scene_bounds = scene_bounds
        self.time_limit = time_limit
        self._step = 0
        self._prev_obs: Observation = None
        self._episode_goal: Goal = None

    @abc.abstractmethod
    def get_demo(self) -> Demo:
        """Provide a demonstration for a current task."""

    @final
    def action_spec(self) -> dm_env.specs.BoundedArray:
        xyz_min, xyz_max = np.split(np.asarray(self.scene_bounds), 2)
        rot_lim = np.array([np.pi, np.pi / 2, np.pi])
        termsig_min, termsig_max = grasp_min, grasp_max = 0, 1
        low = np.r_[xyz_min, -rot_lim, grasp_min, termsig_min]
        high = np.r_[xyz_max, rot_lim, grasp_max, termsig_max]
        return dm_env.specs.BoundedArray(
            minimum=low,
            maximum=high,
            shape=low.shape,
            dtype=np.float32,
            name='[x, y, z, yaw, pitch, roll, grasp, termsig]'
        )

    def observation_spec(self) -> ObservationSpec:
        assert self._prev_obs is not None, 'Env is not initialized. Use .reset once.'

        def np_to_spec(x):
            if isinstance(x, str):
                return dm_env.specs.StringArray(())
            return dm_env.specs.Array(x.shape, x.dtype)
        return tree.map_structure(np_to_spec, self._prev_obs)

    def set_goal(self, text_description: str) -> None:
        """Update episode goal."""
        dt = np.dtype('U77')  # CLIP limit.
        description = np.array(text_description, dtype=dt)
        self._episode_goal = description

    def get_goal(self) -> Goal:
        """Episode goal accessor."""
        return self._episode_goal

    def _as_time_step(self,
                      obs: Observation,
                      reward: float,
                      terminate: bool
                      ) -> dm_env.TimeStep:
        """Wrap variables."""
        if terminate:
            obs = obs.replace(is_terminal=True)
            return dm_env.termination(reward, obs)
        if self._step >= self.time_limit:
            return dm_env.truncation(reward, obs)
        return dm_env.transition(reward, obs)
