import abc
import collections.abc
from typing import TypeAlias, TypedDict

import tree
import numpy as np
import dm_env.specs

Array: TypeAlias = np.ndarray
Goal: TypeAlias = dict[str, Array]
NLGoalKey = 'description'


# Should it be fixed number of fields like in NamedTuple?
class Observation(TypedDict, total=False):
    """Complete sensory perception of an environment."""

    images: list[Array]  # N x (H, W, 3)
    depth_maps: list[Array]  # N x (H, W)
    point_clouds: list[Array]  # N x (H, W, 3)
    joint_velocities: Array
    joint_positions: Array
    tcp_pose: Array
    gripper_pos: Array
    gripper_is_object_detected: Array
    goal: Goal


Action = Array
ActionSpec = dm_env.specs.BoundedArray
ObservationSpec = dict[str, dm_env.specs.Array]
Demo = collections.abc.Sequence[Observation]


class GoalConditionedEnv(abc.ABC):
    """Multi-task, goal-conditioned env.
    Mostly follows dm_env.Environment specification except goal/task sampling.
    """

    def __init__(self,
                 time_limit: int = float('inf')
                 ) -> None:
        self.time_limit = time_limit
        self._step = 0
        self._prev_obs: Observation = None
        self._episode_goal: Goal = None

    @abc.abstractmethod
    def reset(self, task: str) -> dm_env.TimeStep:
        """Reset to the specific task or random one.
        Also update step, prev_obs and episode_goal.
        """

    @abc.abstractmethod
    def step(self, action: Array) -> dm_env.TimeStep:
        """Advance state."""

    @abc.abstractmethod
    def get_demo(self) -> Demo:
        """Provide a demonstration for the current task."""

    @abc.abstractmethod
    def action_spec(self) -> ActionSpec:
        """Define how to interact with the environment."""

    def observation_spec(self) -> ObservationSpec:
        assert self._prev_obs is not None, 'Env is not initialized. Use .reset once.'

        def np_to_spec(x):
            if isinstance(x, str):
                return dm_env.specs.StringArray(())
            return dm_env.specs.Array(x.shape, x.dtype)
        return tree.map_structure(np_to_spec, self._prev_obs)

    def close(self) -> None:
        """Close all streams."""

    def _as_timestep(self,
                     obs: Observation,
                     reward: float,
                     terminate: bool
                     ) -> dm_env.TimeStep:
        if terminate:
            return dm_env.termination(reward, obs)
        if self._step >= self.time_limit:
            return dm_env.truncation(reward, obs)
        return dm_env.transition(reward, obs)


def environment_loop(policy, env):
    ts = env.reset()
    cumsum = 0
    while not ts.last():
        state = env.infer_state(ts.observation)
        action = policy(state)
        ts = env.step(action)
        cumsum += ts.reward
    return cumsum
