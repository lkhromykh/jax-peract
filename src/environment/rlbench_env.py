import logging

import numpy as np
import dm_env.specs

from rlbench import tasks as rlb_tasks
from rlbench.environment import Environment
from rlbench.backend.observation import Observation as rlbObservation
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError

from src.environment import base

Array = np.ndarray
_OBS_CONFIG = ObservationConfig()
_OBS_CONFIG.set_all(True)


class RLBenchEnv(base.GoalConditionedEnv):

    CAMERAS = ('front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist')

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_mode = MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete())
        self.env = Environment(self.action_mode,
                               headless=True,
                               obs_config=_OBS_CONFIG)

    def reset(self, task: str) -> dm_env.TimeStep:
        task = getattr(rlb_tasks, task)
        self.task = self.env.get_task(task)
        self.task.sample_variation()
        descriptions, obs = self.task.reset()
        self._episode_goal = descriptions[0]
        self._prev_obs = self.transform_observation(obs, self._episode_goal)
        self._step = 0
        return dm_env.restart(self._prev_obs)

    def step(self, action: base.Action) -> dm_env.TimeStep:
        action, termsig = np.split(action, [-1])
        try:
            obs, reward, terminate = self.task.step(action)
        except (IKError, InvalidActionError, ConfigurationPathError) as exc:
            logging.info(f'Action {action} led to the exception: {exc}.')
            obs, reward, terminate = self._prev_obs, -1., True
        else:
            obs = self._prev_obs = self.transform_observation(obs, self._episode_goal)
        terminate |= termsig.item() > 0.5
        self._step += 1
        return self._as_timestep(obs, reward, terminate)

    def get_demo(self) -> base.Demo:
        self.task.sample_variation()
        demo = self.task.get_demos(amount=1, live_demos=True)[0]
        descriptions, _ = self.task.reset_to_demo(demo)
        demo = map(lambda obs: self.transform_observation(obs, descriptions[0]), demo)
        return list(demo)

    def close(self) -> None:
        return self.env.shutdown()

    def action_spec(self) -> base.ActionSpec:
        # Termination signal is handled separately.
        size = self.action_mode.action_shape(self.task._scene)
        return dm_env.specs.Array((size + 1,), np.float32)

    # TODO: validate limits
    @classmethod
    def transform_observation(cls, obs: rlbObservation, description: str) -> base.Observation:
        def maybe_append(list_, cam, data):
            if (data := getattr(obs, f'{cam}_{data}')) is not None:
                list_.append(data)

        images, depths, point_cloud = [], [], []
        for cam in cls.CAMERAS:
            maybe_append(images, cam, 'rgb')
            maybe_append(depths, cam, 'depth')
            maybe_append(point_cloud, cam, 'point_cloud')
        # should properly transform an observation.
        def gpos_fn(joints): return 1. - joints.sum(keepdims=True) / 0.08
        def gforce_fn(force): return np.any(force > 0.2)
        identity = lambda x: x
        return base.Observation(
            images=np.stack(images),
            depth_maps=np.stack(depths),
            point_clouds=np.stack(point_cloud),
            joint_velocities=obs.joint_velocities,
            joint_positions=obs.joint_positions,
            tcp_pose=obs.gripper_pose,
            gripper_pos=gpos_fn(obs.gripper_joint_positions),
            gripper_is_object_detected=identity(obs.gripper_touch_forces),
            goal={base.NLGoalKey: description}
        )
