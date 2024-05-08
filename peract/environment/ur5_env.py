import dm_env
import numpy as np
from scipy.spatial.transform import Rotation
from ur_env.remote import RemoteEnvClient

from peract.environment import gcenv


class UREnv(gcenv.GoalConditionedEnv):

    def __init__(self,
                 address: tuple[str, int],
                 scene_bounds: gcenv.SceneBounds,
                 time_limit: int
                 ) -> None:
        super().__init__(scene_bounds=scene_bounds, time_limit=time_limit)
        self._env = RemoteEnvClient(address)

    def reset(self) -> dm_env.TimeStep:
        ts = self._env.reset()
        self._prev_obs = self.extract_observation(ts.observation)
        self._episode_goal = self._prev_obs.goal
        self._step = 0
        return dm_env.restart(self._prev_obs)

    def step(self, action) -> dm_env.TimeStep:
        pos, euler, other = np.split(action, [3, 6])
        rotvec = Rotation.from_euler('ZYX', euler).as_rotvec()
        action = np.r_[pos, rotvec, other].astype(np.float32)
        ts = self._env.step(action)
        self._step += 1
        self._prev_obs = self.extract_observation(ts.observation)
        return self._as_time_step(self._prev_obs, ts.reward, ts.last())

    def get_demo(self) -> gcenv.Demo:
        raise RuntimeError('get_demo is not provided.')

    @staticmethod
    def extract_observation(obs: dict[str, np.ndarray]) -> gcenv.Observation:
        pos, quat = np.split(obs['tcp_pose'], [3])
        euler = Rotation.from_quat(quat).as_euler('ZYX')
        tcp_pose = np.r_[pos, euler]
        def rot_kinect(x): return np.fliplr(np.swapaxes(x, 0, 1)),
        return gcenv.Observation(
            images=rot_kinect(obs['image']),
            depth_maps=rot_kinect(obs['depth']),
            point_clouds=rot_kinect(obs['point_cloud']),
            joint_positions=np.asarray(obs['joint_position']),
            joint_velocities=np.asarray(obs['joint_velocity']),
            tcp_pose=tcp_pose,
            gripper_pos=obs['gripper_pos'],
            gripper_is_obj_detected=obs['gripper_is_obj_detected'],
            is_terminal=obs['is_terminal'],
            goal=obs['description']
        )

    def close(self):
        self._env.close()
