import abc
import collections.abc
from typing import TypeAlias, TypedDict, final

import tree
import numpy as np
import dm_env.specs

Array: TypeAlias = np.ndarray
Goal: TypeAlias = dict[str, Array]
SceneBounds: TypeAlias = tuple[float, float, float, float, float, float]
NLGoalKey = 'description'


class Observation(TypedDict):
    """Complete action-centric environment state representation."""

    images: Array  # N x (H, W, 3), N -- number of cam views.
    depth_maps: Array  # N x (H, W)
    point_clouds: Array  # N x (H, W, 3)
    joint_velocities: Array
    joint_positions: Array
    tcp_pose: Array  # [x, y, z, yaw, pitch, roll]
    gripper_pos: float  # [open=0, close=1]
    gripper_is_obj_detected: bool
    is_terminal: bool
    goal: Goal


Action: TypeAlias = Array
ActionSpec: TypeAlias = dm_env.specs.BoundedArray
ObservationSpec: TypeAlias = dict[str, dm_env.specs.Array]
Demo = collections.abc.Sequence[Observation]


class GoalConditionedEnv(dm_env.Environment):
    """At the moment rather Natural Language conditioned env."""

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
        termsig_min, termsig_max = grip_min, grip_max = 0, 1
        low = np.r_[xyz_min, -rot_lim, grip_min, termsig_min]
        high = np.r_[xyz_max, rot_lim, grip_max, termsig_max]
        return dm_env.specs.BoundedArray(
            minimum=low,
            maximum=high,
            shape=low.shape,
            dtype=np.float32,
            name='[x, y, z, yaw, pitch, roll, grip, termsig]'
        )

    def observation_spec(self) -> ObservationSpec:
        assert self._prev_obs is not None, 'Env is not initialized. Use .reset once.'

        def np_to_spec(x):
            if isinstance(x, str):
                return dm_env.specs.StringArray(())
            return dm_env.specs.Array(x.shape, x.dtype)
        return tree.map_structure(np_to_spec, self._prev_obs)

    def _set_goal(self, text_description: str) -> None:
        """Update episode goal."""
        dt = np.dtype('U77')  # CLIP limit.
        description = np.array(text_description, dtype=dt)
        self._episode_goal = {NLGoalKey: description}

    def _as_timestep(self,
                     obs: Observation,
                     reward: float,
                     terminate: bool
                     ) -> dm_env.TimeStep:
        """Wrap variables."""
        if terminate:
            obs['is_terminal'] = True
            return dm_env.termination(reward, obs)
        if self._step >= self.time_limit:
            return dm_env.truncation(reward, obs)
        return dm_env.transition(reward, obs)
