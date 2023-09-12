import tree
import numpy as np
from rlbench.demo import Demo
from rlbench.backend.observation import Observation

from src.types_ import Trajectory


def is_keyframe(obs: Observation, next_obs: Observation) -> bool:
    predicate = np.all(next_obs.joint_velocities < 1e-2)
    predicate &= obs.gripper_open != next_obs.gripper_open
    return predicate


def extract_trajectory(demo: Demo) -> Trajectory:
    observations, actions = [], []
    keyframe = demo[-1]
    for obs, next_obs in reversed(zip(demo[:-1], demo[1:])):
        if is_keyframe(obs, next_obs):
            keyframe = next_obs
        observations.append(obs)
        actions.append(keyframe)
    return dict(observations=observations, actions=actions)


def as_tfdataset(trajs: list[Trajectory]) -> 'tf.data.Datset':
    # Avoid imports on the top level.
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')

    ds = tree.map_structure(lambda *t: np.concatenate(t), *trajs)
    return tf.data.Dataset.from_tensor_slices(ds)
