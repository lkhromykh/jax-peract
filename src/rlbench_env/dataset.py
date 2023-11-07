from typing import Callable

import tree
import numpy as np
import tensorflow as tf
from rlbench.demo import Demo
from rlbench.backend.observation import Observation

import src.types_ as types


def is_keyframe(obs: Observation, next_obs: Observation) -> bool:
    is_waypoint = obs.gripper_open == next_obs.gripper_open
    is_waypoint &= np.allclose(next_obs.joint_velocities, 0, atol=1e-1)
    is_grasp = obs.gripper_open != next_obs.gripper_open
    return is_waypoint | is_grasp


def extract_trajectory(
        demo: Demo,
        observation_transform: Callable[[Observation], types.Observation],
        action_transform: Callable[[Observation], types.Action]
) -> types.Trajectory:
    observations, actions = [], []
    keyframe = action_transform(demo[-1])
    counter = 0
    for obs, next_obs in reversed(list(zip(demo[:-1], demo[1:]))):
        if is_keyframe(obs, next_obs):
            counter += 1
            keyframe = action_transform(next_obs)
        observations.append(observation_transform(obs))
        actions.append(keyframe)
    print('Keyframes num: ', counter)
    def stack(ts): return tree.map_structure(lambda *t: np.stack(t), *ts)
    observations, actions = map(stack, (observations, actions))
    return types.Trajectory(observations=observations, actions=actions)


def as_tfdataset(trajs: list[types.Trajectory]) -> tf.data.Dataset:
    ds = tree.map_structure(lambda *x: np.concatenate(x), *trajs)
    return tf.data.Dataset.from_tensor_slices(ds)
