from typing import Any

import dm_env
import numpy as np
from scipy.spatial.transform import Rotation as R
from ur_env.remote import RemoteEnvClient

from src.environment import gcenv


class UREnv(gcenv.GoalConditionedEnv):

    def __init__(self,
                 address: tuple[str, int],
                 scene_bounds: gcenv.SceneBounds,
                 ) -> None:
        super().__init__(scene_bounds=scene_bounds, time_limit=float('inf'))
        self._env = RemoteEnvClient(address)

    def reset(self) -> dm_env.TimeStep:
        ts = self._env.reset()
        self._prev_obs = self.extract_observation(ts.observation)
        self._episode_goal = self._prev_obs.goal
        self._step = 0
        return dm_env.restart(self._prev_obs)

    def step(self, action) -> dm_env.TimeStep:
        pos, euler, other = np.split(action, [3, 6])
        rotvec = R.from_euler('ZYX', euler).as_rotvec()
        action = np.r_[pos, rotvec, other].astype(np.float32)
        ts = self._env.step(action)
        self._step += 1
        self._prev_obs = self.extract_observation(ts.observation)
        return ts._replace(observation=self._prev_obs)

    def get_demo(self) -> gcenv.Demo:
        raise RuntimeError('get_demo is not provided.')

    @staticmethod
    def extract_observation(obs: dict[str, Any]) -> gcenv.Observation:
        pos, rotvec = np.split(obs['arm/ActualTCPPose'], [3])
        euler = R.from_rotvec(rotvec).as_euler('ZYX')
        tcp_pose = np.r_[pos, euler]
        def rot_kinect(x): return np.fliplr(np.swapaxes(x, 0, 1)),
        return gcenv.Observation(
            images=rot_kinect(obs['kinect/image']),
            depth_maps=rot_kinect(obs['kinect/depth']),
            point_clouds=rot_kinect(obs['kinect/point_cloud']),
            joint_positions=obs['arm/ActualQ'],
            joint_velocities=obs['arm/ActualQd'],
            tcp_pose=tcp_pose,
            gripper_pos=obs['gripper/pos'],
            gripper_is_obj_detected=obs['gripper/object_detected'],
            is_terminal=obs['is_terminal'],
            goal=obs['description']
        )

    def close(self):
        self._env.close()
