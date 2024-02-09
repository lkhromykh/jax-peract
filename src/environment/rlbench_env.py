import logging

import numpy as np
import dm_env.specs
from scipy.spatial.transform import Rotation

from rlbench import tasks as rlb_tasks
from rlbench.environment import Environment
from rlbench.backend.observation import Observation as rlbObservation
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError

from src.environment import gcenv

Array = np.ndarray
_OBS_CONFIG = ObservationConfig()
_OBS_CONFIG.set_all(True)


class RLBenchEnv(gcenv.GoalConditionedEnv):

    CAMERAS = ('front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist')
    TASKS = ('OpenDrawer',)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_mode = MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete())
        self.env = Environment(self.action_mode,
                               headless=True,
                               obs_config=_OBS_CONFIG)
        self.task = None
        self._in_demo_state = False

    def reset(self) -> dm_env.TimeStep:
        task = np.random.choice(self.TASKS)
        task = getattr(rlb_tasks, task)
        self.task = self.env.get_task(task)
        self.task.sample_variation()
        descriptions, obs = self.task.reset()
        self._set_goal(descriptions[0])
        self._prev_obs = self.transform_observation(obs)
        self._step = 0
        self._in_demo_state = False
        return dm_env.restart(self._prev_obs)

    def step(self, action: gcenv.Action) -> dm_env.TimeStep:
        assert not self._in_demo_state
        pos, euler, grip, termsig = np.split(action, [3, 6, 7])
        quat = Rotation.from_euler('ZYX', euler).as_quat(canonical=True)
        action = np.concatenate([pos, quat, 1. - grip])
        try:
            # Here termsig applied after step, since the simulator itself can detect success.
            obs, reward, terminate = self.task.step(action)
        except (IKError, InvalidActionError, ConfigurationPathError) as exc:
            logging.info(f'Action {action} led to the exception: {exc}.')
            obs, reward, terminate = self._prev_obs, -1., True
        else:
            obs = self.transform_observation(obs)
            terminate |= termsig.item() > 0.5
            self._prev_obs = obs.copy()
        self._step += 1
        return self._as_timestep(obs, reward, terminate)

    def get_demo(self) -> gcenv.Demo:
        assert self.task is not None
        self._in_demo_state = True
        self.task.sample_variation()
        demo = self.task.get_demos(amount=1, live_demos=True)[0]
        descriptions, _ = self.task.reset_to_demo(demo)
        self._set_goal(descriptions[0])
        demo = list(map(self.transform_observation, demo))
        demo[-1]['is_terminal'] = True
        return demo

    def close(self) -> None:
        return self.env.shutdown()

    def transform_observation(self, obs: rlbObservation) -> gcenv.Observation:
        def maybe_append(list_, cam, data):
            if (data := getattr(obs, f'{cam}_{data}')) is not None:
                list_.append(data)

        images, depths, point_cloud = [], [], []
        for cam in self.CAMERAS:
            maybe_append(images, cam, 'rgb')
            maybe_append(depths, cam, 'depth')
            maybe_append(point_cloud, cam, 'point_cloud')
        # should properly transform an observation.
        def gpos_fn(joints): return 1. - np.clip(joints.sum(keepdims=True) / 0.08, 0, 1)
        def gforces_fn(forces): return not np.allclose(forces, 0, atol=0.15)
        pos, quat = np.split(obs.gripper_pose, [3])
        euler = Rotation.from_quat(quat).as_euler('ZYX')
        tcp_pose = np.concatenate([pos, euler])
        return gcenv.Observation(
            images=np.stack(images),
            depth_maps=np.stack(depths),
            point_clouds=np.stack(point_cloud),
            joint_velocities=obs.joint_velocities,
            joint_positions=obs.joint_positions,
            tcp_pose=tcp_pose,
            gripper_pos=gpos_fn(obs.gripper_joint_positions),
            gripper_is_obj_detected=gforces_fn(obs.gripper_touch_forces),
            goal=self._episode_goal,
            is_terminal=False
        )
