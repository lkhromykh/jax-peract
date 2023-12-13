import collections
from typing import Callable

import tree
import numpy as np
import tensorflow as tf
from rlbench.demo import Demo
from rlbench.backend.observation import Observation

import src.types_ as types

Carry = collections.namedtuple('Carry', ('next_obs', 'wayp_buffer', 'total_kframes'))
_Qd_THRESHOLD = 1e-1
_SKIP_EVERY = 3


def keyframe_scan(carry: Carry,
                  obs: Observation,
                  ) -> tuple[Carry, bool]:
    """Determine if the next_observation is a keyframe."""
    next_obs, wp_buffer, total_kframes = carry
    is_waypoint = obs.gripper_open == next_obs.gripper_open
    is_waypoint &= np.allclose(next_obs.joint_velocities, 0, atol=_Qd_THRESHOLD)
    is_waypoint &= wp_buffer > _SKIP_EVERY
    is_grasp = obs.gripper_open != next_obs.gripper_open
    is_keyframe = is_waypoint | is_grasp
    carry = Carry(next_obs=obs,
                  wayp_buffer=0 if is_keyframe else wp_buffer + 1,
                  total_kframes=total_kframes + is_keyframe
                  )
    return carry, is_keyframe


# resample number of episodes between keyframe?
def extract_trajectory(
        demo: Demo,
        observation_transform: Callable[[Observation], types.Observation],
        action_transform: Callable[[Observation], types.Action]
) -> types.Trajectory:
    observations, actions = [], []
    rdemo = reversed(demo)
    last_obs = next(rdemo)
    keyframe = action_transform(last_obs)
    carry = Carry(next_obs=last_obs, wayp_buffer=0, total_kframes=1)
    for obs in rdemo:
        next_obs = carry.next_obs
        carry, is_keyframe = keyframe_scan(carry, obs)
        if is_keyframe:
            keyframe = action_transform(next_obs)
        observations.append(observation_transform(obs))
        actions.append(keyframe)
    print('Keyframes/timesteps num:', carry.total_kframes, '/', len(demo))
    def stack(ts): return tree.map_structure(lambda *t: np.stack(t), *ts)
    def to_f16(x): return x.astype(np.float16) if x.dtype.kind == 'f' else x
    observations, actions = map(stack, (observations, actions))
    observations = tree.map_structure(to_f16, observations)
    return types.Trajectory(observations=observations, actions=actions)


def as_tfdataset(trajs: list[types.Trajectory]) -> tf.data.Dataset:
    ds = tree.map_structure(lambda *x: np.concatenate(x), *trajs)
    return tf.data.Dataset.from_tensor_slices(ds)
