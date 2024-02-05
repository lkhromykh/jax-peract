from typing import Any, Callable
from collections import namedtuple
import logging

import numpy as np

from src.environment import gcenv


Carry = namedtuple('Carry', ('next_obs', 'timestep', 'time_to_next', 'total_kframes'))
ScanFn = Callable[[Carry, gcenv.Observation], tuple[Carry, bool]]
FromObservation = FromKeyframe = Any


def default_scan(
        joint_velocities_thresh: float = 1e-1,
        skip_every: int = 4,
        skip_first: int = 2,
        max_intermediate_frames: int = float('inf')
) -> ScanFn:
    """An ARM-like (2105.14829) keyframe heuristic."""
    def keyframe_scan(carry: Carry,
                      obs: gcenv.Observation,
                      ) -> tuple[Carry, bool]:
        # skip_first added in order to ignore initial acceleration.
        next_obs, timestep, time_to_next, total_kframes = carry
        is_grasp = obs['gripper_is_obj_detected'] != next_obs['gripper_is_obj_detected']
        is_waypoint = not is_grasp
        is_waypoint &= np.allclose(next_obs['joint_velocities'], 0, atol=joint_velocities_thresh)
        is_waypoint &= timestep > skip_first
        is_waypoint &= time_to_next > skip_every
        is_far = time_to_next > max_intermediate_frames
        is_keyframe = is_grasp | is_waypoint | is_far
        carry = Carry(next_obs=obs,
                      timestep=timestep-1,
                      time_to_next=0 if is_keyframe else time_to_next + 1,
                      total_kframes=total_kframes + is_keyframe
                      )
        return carry, is_keyframe

    return keyframe_scan


# resample number of episodes between keyframe?
def extract_keyframes(
        demo: gcenv.Demo,
        scan_fn: ScanFn = default_scan(),
        observation_transform: Callable[[gcenv.Observation], FromObservation] = lambda x: x,
        keyframe_transform: Callable[[gcenv.Observation], FromKeyframe] = lambda x: x,
) -> list[tuple[FromObservation, FromKeyframe]]:
    """Extract observation-keyframe pairs from a demo."""
    rdemo = reversed(demo)
    last_obs = next(rdemo)
    assert last_obs['is_terminal']
    keyframe = keyframe_transform(last_obs)
    carry = Carry(next_obs=last_obs,
                  timestep=len(demo) - 1,
                  time_to_next=0,
                  total_kframes=1)
    pairs = []
    for obs in rdemo:
        assert not obs['is_terminal']
        next_obs = carry.next_obs
        carry, is_keyframe = scan_fn(carry, obs)
        if is_keyframe:
            keyframe = keyframe_transform(next_obs)
        pairs.append((observation_transform(obs), keyframe))
    logging.info(f'Keyframes/timesteps num: {carry.total_kframes} / {len(demo)}')
    return pairs[::-1]
