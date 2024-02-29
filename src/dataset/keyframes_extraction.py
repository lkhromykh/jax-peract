from typing import Any, Callable, NamedTuple, TypeAlias

import numpy as np

from src.environment import gcenv
from src.logger import get_logger


class Carry(NamedTuple):

    next_obs: gcenv.Observation
    time_step: int
    time_to_next_kf: int


ScanFn: TypeAlias = Callable[[Carry, gcenv.Observation], tuple[Carry, bool]]
FromObservation: TypeAlias = Any
FromKeyframe: TypeAlias = Any
KeyframesExtractorOutput: TypeAlias = tuple[list[tuple[FromObservation, FromKeyframe]], list[int]]
KeyframesExtractor: TypeAlias = Callable[[gcenv.Demo], KeyframesExtractorOutput]


def default_scan(carry: Carry, obs: gcenv.Observation, skip_every: int = 4) -> tuple[Carry, bool]:
    """
    An ARM-like (2105.14829) keyframe heuristic.
    Skip_every param may vary depending on demo FPS.
    """
    next_obs, time_step, time_to_next_kf = (c := carry).next_obs, c.time_step, c.time_to_next_kf
    is_grasp_or_release = obs.gripper_is_open ^ next_obs.gripper_is_open
    is_waypoint = next_obs.joints_velocity_is_low
    is_waypoint &= np.all(abs(next_obs.joint_velocities) < abs(obs.joint_velocities))
    is_waypoint &= time_to_next_kf > skip_every
    is_keyframe = is_grasp_or_release | is_waypoint
    carry = Carry(next_obs=obs,
                  time_step=time_step - 1,
                  time_to_next_kf=0 if is_keyframe else time_to_next_kf + 1,
                  )
    return carry, is_keyframe


def extractor_factory(
        scan_fn: ScanFn = default_scan,
        observation_transform: Callable[[gcenv.Observation], FromObservation] = lambda x: x,
        keyframe_transform: Callable[[gcenv.Observation], FromKeyframe] = lambda x: x,
) -> KeyframesExtractor:
    """Closure of a function that detects important observations within a demo
    and pairs observations with the next keyframes."""

    def extract_keyframes(demo: gcenv.Demo) -> KeyframesExtractorOutput:
        rdemo = reversed(demo)
        last_obs = next(rdemo)
        assert last_obs.is_terminal, 'Demo is truncated or unsuccessful.'
        carry = Carry(next_obs=last_obs,
                      time_step=len(demo) - 1,
                      time_to_next_kf=0)
        pairs, kf_time_steps = [], []
        is_keyframe = last_obs.is_terminal  # == True
        for obs in rdemo:
            assert not obs.is_terminal, 'Mid episode termination.'
            if is_keyframe:
                kf_time_steps.append(carry.time_step)
                keyframe = keyframe_transform(carry.next_obs)
            pairs.append((observation_transform(obs), keyframe))
            carry, is_keyframe = scan_fn(carry, obs)
        kf_time_steps = kf_time_steps[::-1]
        get_logger().info('Task %s; Keyframes time steps: %s',  last_obs.goal, kf_time_steps)
        return pairs[::-1], kf_time_steps

    return extract_keyframes
