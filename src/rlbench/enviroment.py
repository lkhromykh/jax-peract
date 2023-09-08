import tree
import numpy as np
import dm_env.specs

from rlbench import tasks
from rlbench.environment import Environment
from rlbench.backend.observation import Observation
from rlbench.observation_config import ObservationConfig

from src.rlbench.action_mode import ActionMode
from src.rlbench.voxel_grid import VoxelGrid
from src import types_ as types

Array = types.Array


class RLBenchEnv(dm_env.Environment):

    def __init__(self,
                 task: str,
                 action_bounds: tuple[Array, Array],
                 nbins: int,
                 obs_config: ObservationConfig = ObservationConfig()
                 ) -> None:
        self.action_mode = ActionMode(action_bounds, nbins)
        self.env = Environment(self.action_mode,
                               headless=False,
                               obs_config=obs_config)
        self.task = self.env.get_task(getattr(tasks, task))
        scene_bounds = tuple(map(lambda x: x[:3], action_bounds))
        self._vgrid = VoxelGrid(scene_bounds, nbins)

    def reset(self) -> dm_env.TimeStep:
        description, obs = self.task.reset()
        obs = self._transform_observation(obs)
        return dm_env.restart(obs)

    def step(self, action: Array) -> dm_env.TimeStep:
        obs, reward, terminate = self.task.step(action)
        obs = self._transform_observation(obs)
        if terminate:
            return dm_env.termination(reward, obs)
        return dm_env.transition(reward, obs)

    def action_spec(self):
        return self.action_mode.action_spec()

    def observation_spec(self) -> types.ObservationSpec:
        _, obs = self.task.reset()
        def convert(x): return dm_env.specs.Array(x.shape, x.dtype)
        obs = self._transform_observation(obs)
        return tree.map_structure(convert, obs)

    def _transform_observation(self, obs: Observation) -> types.Observation:
        return obs
        lowdim = obs.get_low_dim_data()
        voxels = self._vgrid(obs)
        return dict(lowdim=lowdim, voxels=voxels)

    def get_demos(self, amount: int) -> list['Trajectory']:
        raw_demos = self.task.get_demos(amount=amount, live_demos=True)
        demos = []
        for raw_demo in raw_demos:
            demo = map(self._transform_observation, raw_demo)
            demo = tree.map_structure(lambda *t: np.stack(t), *demo)
            demos.append(demo)
        return demos

    def close(self) -> None:
        self.env.shutdown()
