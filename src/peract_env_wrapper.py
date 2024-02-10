from typing import NamedTuple

import numpy as np
import dm_env.specs

from src import utils
from src.environment import gcenv
import src.types_ as types


class PerActEncoders(NamedTuple):
    """Extract information from an action-centric observation."""

    scene_encoder: utils.VoxelGrid
    action_encoder: utils.DiscreteActionTransform
    text_encoder: utils.CLIP

    def infer_state(self, obs: gcenv.Observation) -> types.State:
        low_dim = np.atleast_1d(obs['gripper_pos'], obs['gripper_is_obj_detected'])
        low_dim = np.concatenate(low_dim)
        return types.State(
            voxels=self.scene_encoder.encode(obs),
            low_dim=low_dim,
            goal=self.text_encoder.encode(obs['goal'][gcenv.NLGoalKey])
        )

    def infer_action(self, obs: gcenv.Observation) -> types.Action:
        action = np.r_[obs['tcp_pose'], obs['gripper_is_obj_detected'], obs['is_terminal']]
        return self.action_encoder.encode(action)

    def observation_spec(self) -> types.State:
        return types.State(
            voxels=self.scene_encoder.observation_spec(),
            low_dim=dm_env.specs.Array((2,), np.float32),
            goal=self.text_encoder.observation_spec(),
        )

    def action_spec(self) -> types.ActionSpec:
        return self.action_encoder.action_spec()


class PerActEnvWrapper(dm_env.Environment):
    """Preprocess generic env observations to the PerAct state."""

    def __init__(self,
                 env: gcenv.GoalConditionedEnv,
                 encoders: PerActEncoders
                 ) -> None:
        self.env = env
        self.encoders = encoders

    def reset(self) -> dm_env.TimeStep:
        ts = self.env.reset()
        state = self.encoders.infer_state(ts.observation)
        return ts._replace(observation=state)

    def step(self, action: gcenv.Action) -> dm_env.TimeStep:
        action = self.encoders.action_encoder.decode(action)
        ts = self.env.step(action)
        return ts._replace(observation=self.encoders.infer_state(ts.observation))

    def action_spec(self) -> types.ActionSpec:
        return self.encoders.action_spec()

    def observation_spec(self):
        return self.encoders.observation_spec()
