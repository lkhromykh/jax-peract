import tree
import numpy as np
import dm_env.specs

from rlbench import tasks
from rlbench.environment import Environment
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError

from src.rlbench.action_mode import ActionMode
from src.rlbench.voxel_grid import VoxelGrid
import src.rlbench.dataset as rlds
from src import types_ as types

Array = types.Array

_OBS_CONFIG = ObservationConfig()
_OBS_CONFIG.gripper_touch_forces = True


class RLBenchEnv(dm_env.Environment):

    def __init__(self,
                 task: str,
                 scene_bounds: tuple[Array, Array],
                 nbins: int,
                 obs_config: ObservationConfig = _OBS_CONFIG
                 ) -> None:
        scene_bounds = tuple(map(np.asanyarray, scene_bounds))
        self.action_mode = ActionMode(scene_bounds, nbins)
        self.env = Environment(self.action_mode,
                               headless=True,
                               obs_config=obs_config)
        self.task = self.env.get_task(getattr(tasks, task))
        self.vgrid = VoxelGrid(scene_bounds, nbins)

    def reset(self) -> dm_env.TimeStep:
        description, obs = self.task.reset()
        obs = self._transform_observation(obs)
        self._prev_obs = obs
        return dm_env.restart(obs)

    def step(self, action: Array) -> dm_env.TimeStep:
        try:
            obs, reward, terminate = self.task.step(action)
            obs = self._transform_observation(obs)
            self._prev_obs = obs
            if terminate:
                return dm_env.termination(reward, obs)
            return dm_env.transition(reward, obs)
        except (IKError, InvalidActionError):
            return dm_env.termination(0., self._prev_obs)

    def action_spec(self) -> types.ActionSpec:
        return self.action_mode.action_spec()

    def observation_spec(self) -> types.ObservationSpec:
        _, obs = self.task.reset()
        obs = self._transform_observation(obs)
        def convert(x): return dm_env.specs.Array(x.shape, x.dtype)
        return tree.map_structure(convert, obs)

    def _transform_observation(self, obs: Observation) -> types.Observation:
        voxels = self.vgrid(obs)
        low_dim = obs.get_low_dim_data()
        return types.Observation(voxels=voxels, low_dim=low_dim)

    def get_demos(self, amount: int) -> list[types.Trajectory]:
        trajs = []
        raw_demos = self.task.get_demos(amount=amount, live_demos=True)
        for demo in raw_demos:
            traj = rlds.extract_trajectory(
                demo=demo,
                observation_transform=self._transform_observation,
                action_transform=self.action_mode.from_observation
            )
            trajs.append(traj)
        return trajs

    def close(self) -> None:
        return self.env.shutdown()
