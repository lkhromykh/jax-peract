import dm_env
import numpy as np

from ur_env.remote import RemoteEnvClient

from src.environment import base

Array = np.ndarray
Obs = dict[str, Array]


class UR5e(base.GoalConditionedEnv):

    def __init__(self, address, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = RemoteEnvClient(address)

    def reset(self, task: str) -> dm_env.TimeStep:
        self._episode_goal = {base.NLGoalKey: task}
        return self.env.reset()

    def transform_observation(self, obs: Obs) -> Obs:
        return obs | self._epsode_goal

    def __getattr__(self, item):
        return getattr(self.env, item)
