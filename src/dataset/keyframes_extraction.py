from typing import Any, Callable, NamedTuple, TypeAlias
import logging

import numpy as np

from src.environment import gcenv


class Carry(NamedTuple):
    next_obs: gcenv.Observation
    timestep: int
    time_to_next_kf: int


ScanFn: TypeAlias = Callable[[Carry, gcenv.Observation], tuple[Carry, bool]]
FromObservation: TypeAlias = Any
FromKeyframe: TypeAlias = Any
KeyframesExtractorOutput: TypeAlias = tuple[list[tuple[FromObservation, FromKeyframe]], list[int]]
KeyframesExtractor: TypeAlias = Callable[[gcenv.Demo], KeyframesExtractorOutput]


def default_scan_factory(
        joint_velocities_thresh: float = 0.1,
        skip_every: int = 4,
) -> ScanFn:
    """An ARM-like (2105.14829) keyframe heuristic."""
    def keyframe_scan(carry: Carry,
                      obs: gcenv.Observation,
                      ) -> tuple[Carry, bool]:
        # skip_first added in order to ignore initial acceleration.
        next_obs, timestep, time_to_next_kf = carry
        is_grasp = obs['gripper_is_obj_detected'] != next_obs['gripper_is_obj_detected']
        is_decelerating = np.all(abs(next_obs['joint_velocities']) < abs(obs['joint_velocities']))
        is_waypoint = not is_grasp
        is_waypoint &= is_decelerating
        is_waypoint &= np.allclose(next_obs['joint_velocities'], 0, atol=joint_velocities_thresh)
        is_waypoint &= time_to_next_kf > skip_every
        is_keyframe = is_grasp | is_waypoint
        carry = Carry(next_obs=obs,
                      timestep=timestep - 1,
                      time_to_next_kf=0 if is_keyframe else time_to_next_kf + 1,
                      )
        return carry, is_keyframe

    return keyframe_scan


def extractor_factory(
        scan_fn: ScanFn = default_scan_factory(),
        observation_transform: Callable[[gcenv.Observation], FromObservation] = lambda x: x,
        keyframe_transform: Callable[[gcenv.Observation], FromKeyframe] = lambda x: x,
) -> KeyframesExtractor:
    """Closure of a function that detects important observations within a demo
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
                      time_to_next_kf=0)
        pairs, kf_idxs = [], [carry.timestep]
        for obs in rdemo:
            next_obs = carry.next_obs
            carry, is_keyframe = scan_fn(carry, obs)
            if is_keyframe:
                kf_idxs.append(carry.timestep)
                keyframe = keyframe_transform(next_obs)
            pairs.append((observation_transform(obs), keyframe))
        kf_idxs = kf_idxs[::-1]
        logging.info('Keyframes indices: %s',  kf_idxs)
        return pairs[::-1], kf_idxs

    return extract_keyframes
