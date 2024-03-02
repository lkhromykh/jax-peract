from typing import Any, Callable, NamedTuple, TypeAlias

import numpy as np

from src.environment import gcenv
from src.logger import get_logger


class Carry(NamedTuple):

    observation: gcenv.Observation
    time_step: int
    time_to_next_kf: int


ScanFn: TypeAlias = Callable[[Carry, gcenv.Observation], tuple[Carry, bool]]
FromObservation: TypeAlias = Any
FromKeyframe: TypeAlias = Any
KeyframesExtractorOutput: TypeAlias = tuple[list[tuple[FromObservation, FromKeyframe]], list[int]]
KeyframesExtractor: TypeAlias = Callable[[gcenv.Demo], KeyframesExtractorOutput]


def default_scan(carry_tp1: Carry, obs_t: gcenv.Observation, skip_every: int = 4) -> tuple[Carry, bool]:
    """
    An ARM-like (2105.14829) keyframe heuristic.
    Skip_every param may vary depending on demo FPS.
    """
    obs_tp1, time_step, time_to_next_kf = (c := carry_tp1).observation, c.time_step, c.time_to_next_kf
    is_grasp_or_release = obs_t.gripper_is_open ^ obs_tp1.gripper_is_open
    is_waypoint = obs_tp1.joint_velocities_are_low
    is_waypoint &= np.all(abs(obs_tp1.joint_velocities) < abs(obs_t.joint_velocities))
    is_waypoint &= time_to_next_kf > skip_every
    is_keyframe = is_grasp_or_release | is_waypoint
    carry_t = Carry(
        observation=obs_t,
        time_step=time_step - 1,
        time_to_next_kf=0 if is_keyframe else time_to_next_kf + 1,
    )
    return carry_t, is_keyframe


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
        carry = Carry(observation=last_obs,
                      time_step=len(demo) - 1,
                      time_to_next_kf=0)
        pairs, kf_time_steps = [], []
        next_carry, is_keyframe = carry, last_obs.is_terminal  # == True
        for obs in rdemo:
            assert not obs.is_terminal, 'Mid episode termination.'
            if is_keyframe:
                kf_time_steps.append(next_carry.time_step)
                keyframe = keyframe_transform(next_carry.observation)
            pairs.append((observation_transform(obs), keyframe))
            next_carry = carry
            carry, is_keyframe = scan_fn(carry, obs)
        kf_time_steps = kf_time_steps[::-1]
        get_logger().info('Task %s; Keyframes time steps: %s',  last_obs.goal, kf_time_steps)
        return pairs[::-1], kf_time_steps

    return extract_keyframes
