import numpy as np
import dm_env.specs

from src.environment import base, encoders
import src.types_ as types


class PerActWrapper:

    def __init__(self,
                 env: base.GoalConditionedEnv | None,
                 scene_encoder: encoders.VoxelGrid,
                 action_encoder: encoders.DiscreteActionTransform,
                 text_tokenizer: encoders.TextTokenizer
                 ) -> None:
        self.env = env
        self.scene_encoder = scene_encoder
        self.action_encoder = action_encoder
        self.text_tokenizer = text_tokenizer

    def reset(self, task: str) -> dm_env.TimeStep:
        assert self.is_headless()
        ts = self.env.reset(task)
        return ts._replace(observation=self.infer_state(ts.observation))

    def step(self, action: base.Action) -> dm_env.TimeStep:
        assert self.is_headless()
        action = self.action_encoder.decode(action)
        ts = self.env.step(action)
        return ts._replace(observation=self.infer_state(ts.observation))

    def infer_action(self, obs: base.Observation) -> base.Action:
        """Use action-centric representation to infer action that corresponds to the obs."""
        action = np.concatenate([obs['tcp_pose'], obs['gripper_is_object_detected']])
        return self.action_encoder.encode(action)

    def infer_state(self, obs: base.Observation) -> types.State:
        low_dim = np.atleast_1d(obs['gripper_is_object_detected'])
        return types.State(
            voxels=self.scene_encoder.encode(obs),
            low_dim=low_dim,
            goal=self.text_tokenizer.encode(obs['goal'])
        )

    def action_spec(self) -> list[dm_env.specs.DiscreteArray]:
        return self.action_encoder.action_spec()

    def is_headless(self) -> bool:
        return self.env is None

    def __getattr__(self, item):
        return getattr(self.env, item)

    @classmethod
    def from_params(cls,
                    max_text_length: int,
                    scene_bounds: tuple[float, float, float, float, float, float],
                    scene_bins: int,
                    rot_bins: int,
                    rot_repr: str
                    ) -> 'PerActWrapper':
        return ...

