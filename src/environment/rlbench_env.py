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
from src.logger import get_logger

Array = np.ndarray
_OBS_CONFIG = ObservationConfig()
_OBS_CONFIG.set_all(True)

_CAMERAS = ('front', 'left_shoulder', 'right_shoulder', 'overhead', 'wrist')
EASY_TASKS = (
    'SlideBlockToTarget', 'TurnTap', 'StackBlocks'
    'OpenDrawer', 'PushButton', 'ReachAndDrag',
)
MEDIUM_TASKS = (
    'CloseJar', 'LightBulbIn', 'InsertOntoSquarePeg',
    'PutMoneyInSafe', 'StackWine',
)
HARD_TASKS = (
    'StackCups', 'PutGroceriesInCupboard', 'TakeItemOutOfDrawer',
)


class RLBenchEnv(gcenv.GoalConditionedEnv):

    CAMERAS = _CAMERAS
    TASKS = EASY_TASKS + MEDIUM_TASKS

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
        self.set_goal(descriptions[0])
        self._prev_obs = self.transform_observation(obs)
        self._step = 0
        self._in_demo_state = False
        return dm_env.restart(self._prev_obs)

    def step(self, action: gcenv.Action) -> dm_env.TimeStep:
        assert not self._in_demo_state
        self._step += 1
        pos, euler, grasp, termsig = np.split(action, [3, 6, 7])
        # if termsig > 0.5:
        #     sim_success, sim_terminate = self.task._task.success()
        #     get_logger().info('Is terminating correctly: %s', sim_terminate)
        #     return self._as_time_step(self._prev_obs, float(sim_success), True)
        quat = Rotation.from_euler('ZYX', euler).as_quat(canonical=True)
        action = np.concatenate([pos, quat, 1. - grasp])
        try:
            obs, reward, terminate = self.task.step(action)
            # reward, terminate = 0, False  # ground truth sim state is hidden from an agent until termsig.
        except (IKError, InvalidActionError, ConfigurationPathError) as exc:
            get_logger().info('Action %s led to exception: %s.', action, exc)
            obs, reward, terminate = self._prev_obs, -1., True
        else:
            obs = self.transform_observation(obs)
            self._prev_obs = obs
        return self._as_time_step(obs, reward, terminate)

    def get_demo(self) -> gcenv.Demo:
        assert self.task is not None
        self._in_demo_state = True
        self.task.sample_variation()
        demo = self.task.get_demos(amount=1, live_demos=True)[0]
        descriptions, _ = self.task.reset_to_demo(demo)
        self.set_goal(descriptions[0])
        demo = list(map(self.transform_observation, demo))
        demo[-1] = demo[-1].replace(is_terminal=True)
        return demo

    def close(self) -> None:
        return self.env.shutdown()

    def transform_observation(self, obs: rlbObservation) -> gcenv.Observation:
        def maybe_append(list_, cam, data):
            if (data := getattr(obs, f'{cam}_{data}')) is not None:
                list_.append(data)

        images, depths, point_clouds = [], [], []
        for cam in self.CAMERAS:
            maybe_append(images, cam, 'rgb')
            maybe_append(depths, cam, 'depth')
            maybe_append(point_clouds, cam, 'point_cloud')

        def gpos_fn(joints): return 1. - np.clip(joints.sum(keepdims=True) / 0.08, 0, 1)  # Franka Panda.
        def gforces_fn(forces): return not np.allclose(forces, 0, atol=0.1)
        pos, quat = np.split(obs.gripper_pose, [3])
        euler = Rotation.from_quat(quat).as_euler('ZYX')
        tcp_pose = np.concatenate([pos, euler])
        return gcenv.Observation(
            images=tuple(images),
            depth_maps=tuple(depths),
            point_clouds=tuple(point_clouds),
            joint_velocities=obs.joint_velocities,
            joint_positions=obs.joint_positions,
            tcp_pose=tcp_pose,
            gripper_pos=gpos_fn(obs.gripper_joint_positions),
            gripper_is_obj_detected=gforces_fn(obs.gripper_touch_forces),
            goal=self.get_goal(),
            is_terminal=False
        )
