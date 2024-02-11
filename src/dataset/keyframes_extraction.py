from typing import Any, Callable, NamedTuple, TypeAlias
import logging

import numpy as np

from src.environment import gcenv


class Carry(NamedTuple):

    next_obs: gcenv.Observation
    time_step: int
    time_to_next_kf: int


ScanFn: TypeAlias = Callable[[Carry, gcenv.Observation], tuple[Carry, bool]]
FromObservation: TypeAlias = Any
FromKeyframe: TypeAlias = Any
KeyframesExtractorOutput: TypeAlias = tuple[list[tuple[FromObservation, FromKeyframe]], list[int]]
KeyframesExtractor: TypeAlias = Callable[[gcenv.Demo], KeyframesExtractorOutput]


def default_scan_factory(skip_every: int = 4) -> ScanFn:
    """An ARM-like (2105.14829) keyframe heuristic.
    Skip_every param may vary depending on demo FPS.
    """
    def keyframe_scan(carry: Carry, obs: gcenv.Observation) -> tuple[Carry, bool]:
        # skip_first added in order to ignore initial acceleration.
        next_obs, time_step, time_to_next_kf = (c := carry).next_obs, c.time_step, c.time_to_next_kf
        is_grasp = obs.gripper_is_open ^ next_obs.gripper_is_open
        is_waypoint = next_obs.joints_velocity_is_low
        is_waypoint &= np.all(abs(next_obs.joint_velocities) < abs(obs.joint_velocities))
        is_waypoint &= time_to_next_kf > skip_every
        is_keyframe = is_grasp | is_waypoint
        carry = Carry(next_obs=obs,
                      time_step=time_step - 1,
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
    and pairs observations with next keyframes."""

    def extract_keyframes(demo: gcenv.Demo) -> KeyframesExtractorOutput:
        rdemo = reversed(demo)
        last_obs = next(rdemo)
        assert last_obs.is_terminal, 'Demo is probably truncated or unsuccessful.'
        keyframe = keyframe_transform(last_obs)
        pairs = [(observation_transform(last_obs), keyframe)]  # predict termsig from the state.
        carry = Carry(next_obs=last_obs.replace(is_terminal=False),
                      time_step=len(demo) - 1,
                      time_to_next_kf=0)
        kf_time_steps = [carry.time_step]
        for obs in rdemo:
            assert not obs.is_terminal
            next_obs = carry.next_obs
            carry, is_keyframe = scan_fn(carry, obs)
            if is_keyframe:
                kf_time_steps.append(carry.time_step)
                keyframe = keyframe_transform(next_obs)
            pairs.append((observation_transform(obs), keyframe))
        kf_time_steps = kf_time_steps[::-1]
        breakpoint()
        logging.info('Keyframes time steps: %s',  kf_time_steps)
        return pairs[::-1], kf_time_steps

    return extract_keyframes
