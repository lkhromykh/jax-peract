from enum import IntEnum
from typing import Type
import logging

import tree
import numpy as np
import dm_env.specs

from rlbench import tasks
from rlbench.environment import Environment
from rlbench.backend.task import Task as rlbTask
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError

from src.rlbench_env.action_mode import DiscreteActionMode
from src.rlbench_env.voxel_grid import VoxelGrid
from src.rlbench_env.dataset import extract_trajectory
from src.networks.pretrained import TextEncoder
from src import types_ as types

Array = np.ndarray
_OBS_CONFIG = ObservationConfig()
_OBS_CONFIG.set_all(True)


class Task(IntEnum):

    ReachTarget = 0
    # PickAndLift = 0

    def as_one_hot(self) -> Array:
        task = np.zeros(len(Task), dtype=np.int32)
        task[self] = 1
        return task

    def as_rlbench_task(self) -> Type[rlbTask]:
        return getattr(tasks, self.name)

    @staticmethod
    def sample(rng: np.random.RandomState) -> 'Task':
        return Task(rng.choice(Task))


# TODO: add timestep to the observation? terminate_episode, collision?
class RLBenchEnv(dm_env.Environment):

    def __init__(self,
                 seed: int | np.random.RandomState,
                 scene_bounds: tuple[float, ...],
                 scene_bins: int,
                 rot_bins: int,
                 text_emb_length: int,
                 time_limit: int = float('inf'),
                 ) -> None:
        self.rng = np.random.default_rng(seed)
        self.time_limit = time_limit
        scene_bounds = np.split(np.asarray(scene_bounds, np.float32), 2)
        self.action_mode = DiscreteActionMode(scene_bounds, scene_bins, rot_bins)
        self.env = Environment(self.action_mode,
                               headless=True,
                               shaped_rewards=False,
                               obs_config=_OBS_CONFIG,
                               static_positions=True
                               )
        self.vgrid = VoxelGrid(scene_bounds, scene_bins)
        if text_emb_length > 0:
            self.text_encoder = TextEncoder(max_length=text_emb_length)
        else:
            self.text_encoder = None
        self.reset()  # launch PyRep, init_all attributes.

    def reset(self) -> dm_env.TimeStep:
        task = Task.sample(self.rng)
        self.task = self.env.get_task(task.as_rlbench_task())
        self.text_descriptions, obs = self.task.reset()
        if self.text_encoder is not None:
            self._description = self.text_encoder(self.text_descriptions[0])
        else:
            task_code = task.as_one_hot()
            self._description = np.expand_dims(task_code, 0)
        self._prev_obs = self._transform_observation(obs)
        self._step = 0
        return dm_env.restart(self._prev_obs)

    def step(self, action: Array) -> dm_env.TimeStep:
        try:
            obs, reward, terminate = self.task.step(action)
        except (IKError, InvalidActionError, ConfigurationPathError) as exc:
            logging.info(f'Action {action} led to the exception: {exc}.')
            obs, reward, terminate = self._prev_obs, -1., True
        else:
            obs = self._prev_obs = self._transform_observation(obs)
        self._step += 1
        if terminate:
            return dm_env.termination(reward, obs)
        if self._step >= self.time_limit:
            return dm_env.truncation(reward, obs)
        return dm_env.transition(reward, obs)

    def action_spec(self) -> types.ActionSpec:
        return self.action_mode.action_spec()

    def observation_spec(self) -> types.ObservationSpec:
        def np_to_spec(x): return dm_env.specs.Array(x.shape, x.dtype)
        return tree.map_structure(np_to_spec, self._prev_obs)

    def _transform_observation(self, obs: Observation) -> types.Observation:
        low_dim = np.atleast_1d(obs.gripper_open).astype(np.float32)
        return types.Observation(voxels=self.vgrid(obs),
                                 low_dim=low_dim,
                                 task=self._description
                                 )

    def get_demos(self, amount: int) -> list[types.Trajectory]:
        trajs = []
        for i in range(amount):
            print(f'Traj {i}. ', end='')
            self.reset()  # resample task, update description.
            demo = self.task.get_demos(amount=1, live_demos=True)[0]
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
