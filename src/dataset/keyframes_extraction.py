from typing import Any, Callable, TypeAlias
from collections import namedtuple
import logging

import numpy as np

from src.environment import gcenv


Carry = namedtuple('Carry', ('next_obs', 'timestep', 'time_to_next', 'total_kframes'))
ScanFn: TypeAlias = Callable[[Carry, gcenv.Observation], tuple[Carry, bool]]
FromObservation: TypeAlias = Any
FromKeyframe: TypeAlias = Any
KeyframesExtractorOutput: TypeAlias = tuple[list[tuple[FromObservation, FromKeyframe], list[int]]]
KeyframesExtractor: TypeAlias = Callable[[gcenv.Demo], KeyframesExtractorOutput]


# TODO: check keyframes extraction.
def default_scan_factory(
        joint_velocities_thresh: float = 0.1,
        skip_every: int = 4,
        skip_first: int = 2,
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
        is_keyframe = is_grasp | is_waypoint
        carry = Carry(next_obs=obs,
                      timestep=timestep - 1,
                      time_to_next=0 if is_keyframe else time_to_next + 1,
                      total_kframes=total_kframes + is_keyframe
                      )
        return carry, is_keyframe

    return keyframe_scan


def extractor_factory(
        scan_fn: ScanFn = default_scan_factory(),
        observation_transform: Callable[[gcenv.Observation], FromObservation] = lambda x: x,
        keyframe_transform: Callable[[gcenv.Observation], FromKeyframe] = lambda x: x,
) -> KeyframesExtractor:
    """Closure of a function that detects important observations within demo
    and pairs observation with a next keyframe."""

    def extract_keyframes(
            demo: gcenv.Demo,
    ) -> KeyframesExtractorOutput:
        rdemo = reversed(demo)
        last_obs = next(rdemo)
        assert last_obs['is_terminal']
        keyframe = keyframe_transform(last_obs)
        carry = Carry(next_obs=last_obs,
                      timestep=len(demo) - 1,
                      time_to_next=0,
                      total_kframes=1)
        pairs, kf_idxs = [], [carry.timestep]
        for obs in rdemo:
            assert not obs['is_terminal']
            next_obs = carry.next_obs
            carry, is_keyframe = scan_fn(carry, obs)
            if is_keyframe:
                kf_idxs.append(carry.timestep)
                keyframe = keyframe_transform(next_obs)
            pairs.append((observation_transform(obs), keyframe))
        kf_idxs = kf_idxs[::-1]
        logging.info(f'Keyframes/timesteps num: {carry.total_kframes}/{len(demo)}: {kf_idxs}')
        return pairs[::-1], kf_idxs

    return extract_keyframes
