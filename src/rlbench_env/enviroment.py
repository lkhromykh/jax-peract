from enum import IntEnum
import logging

import tree
import numpy as np
import dm_env.specs

from rlbench import tasks as rlb_tasks
from rlbench.environment import Environment
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError

from src.rlbench_env.action_mode import DiscreteActionMode
from src.rlbench_env.voxel_grid import VoxelGrid
from src.rlbench_env.dataset import extract_trajectory
from src import types_ as types

Array = types.Array
_OBS_CONFIG = ObservationConfig()
_OBS_CONFIG.gripper_touch_forces = True
_OBS_CONFIG.task_low_dim_state = True


class Task(IntEnum):

    ReachTarget = 0
    # PickAndLift = 0

    def as_one_hot(self):
        task = np.zeros(len(Task), dtype=np.int32)
        task[self] = 1
        return task

    def as_rlbench_task(self):
        return getattr(rlb_tasks, self.name)

    @staticmethod
    def sample(rng: np.random.RandomState) -> 'Task':
        return Task(rng.choice(Task))


class RLBenchEnv(dm_env.Environment):

    def __init__(self,
                 rng: np.random.RandomState,
                 scene_bounds: tuple[Array, Array],
                 time_limit: int = float('inf'),
                 obs_config: ObservationConfig = _OBS_CONFIG
                 ) -> None:
        self.rng = np.random.default_rng(rng)
        self.time_limit = time_limit
        scene_bounds = tuple(map(np.asanyarray, scene_bounds))
        self.action_mode = DiscreteActionMode(scene_bounds)
        self.env = Environment(self.action_mode,
                               headless=True,
                               shaped_rewards=False,
                               obs_config=obs_config,
                               )
        self.vgrid = VoxelGrid(scene_bounds, self.action_mode.SCENE_BINS)
        self.reset()  # launch PyRep, init_all attributes.

    def reset(self) -> dm_env.TimeStep:
        task = Task.sample(self.rng)
        self.task = self.env.get_task(task.as_rlbench_task())
        self.description, obs = self.task.reset()
        self.description = task.as_one_hot()  # ignore text description for now.
        obs = self._prev_obs = self._transform_observation(obs)
        self._steps = 0
        return dm_env.restart(obs)

    def step(self, action: Array) -> dm_env.TimeStep:
        try:
            obs, reward, terminate = self.task.step(action)
        except (IKError, InvalidActionError, ConfigurationPathError) as exc:
            logging.info(exc)
            obs, reward, terminate = self._prev_obs, -100., True
        else:
            obs = self._prev_obs = self._transform_observation(obs)
        self._steps += 1
        if terminate:
            return dm_env.termination(reward, obs)
        if self._steps >= self.time_limit:
            return dm_env.truncation(reward, obs)
        return dm_env.transition(reward, obs)

    def action_spec(self) -> types.ActionSpec:
        return self.action_mode.action_spec()

    def observation_spec(self) -> types.ObservationSpec:
        def convert(x): return dm_env.specs.Array(x.shape, x.dtype)
        return tree.map_structure(convert, self._prev_obs)

    def _transform_observation(self, obs: Observation) -> types.Observation:
        voxels = self.vgrid(obs)
        # low_dim = obs.get_low_dim_data()
        low_dim = obs.task_low_dim_state
        return types.Observation(voxels=voxels,
                                 low_dim=low_dim,
                                 task=self.description
                                 )

    def get_demos(self, amount: int) -> list[types.Trajectory]:
        trajs = []
        raw_demos = self.task.get_demos(amount=amount, live_demos=True)
        for demo in raw_demos:
            traj = extract_trajectory(
                demo=demo,
                observation_transform=self._transform_observation,
                action_transform=self.action_mode.from_observation
            )
            trajs.append(traj)
        return trajs

    def close(self) -> None:
        return self.env.shutdown()


def environment_loop(policy, env):
    ts = env.reset()
    cumsum = 0
    while not ts.last():
        action = policy(ts.observation)
        ts = env.step(action)
        cumsum += ts.reward
    return cumsum
