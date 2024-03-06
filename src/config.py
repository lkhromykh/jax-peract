import dataclasses
from typing import TypeAlias

from ruamel.yaml import YAML

Layers: TypeAlias = tuple[int, ...]


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Config:
    # Conv stem
    conv_stem_features: Layers = (64, 64)
    conv_stem_kernels: Layers = (4, 1)
    conv_stem_strides: Layers = (4, 1)
    conv_stem_use_skip_connections: bool = True
    # Perceiver
    latent_dim: int = 512
    latent_channels: int = 512
    num_blocks: int = 1
    num_self_attend_per_block: int = 6
    num_cross_attend_heads: int = 1
    num_self_attend_heads: int = 8
    cross_attend_widening_factor: float = 1.
    self_attend_widening_factor: float = 1.
    use_layer_norm: bool = True
    use_trainable_pos_encoding: bool = False
    prior_initial_scale: float = 0.02
    ff_num_bands: int = 0
    text_context_length: int = 77  # max. 77
    # Action decoder
    act_decoder_mlp_dim: int = 256
    act_decoder_conv_kernel: int = 3
    # Training
    max_grad_norm: float = 10.
    warmup_steps: int = -1
    peak_learning_rate: float = 5e-4
    training_steps: int = 200_000
    batch_size: int = 32
    weight_decay: float = 1e-6
    log_every: int = 500
    save_every: int = 5000
    jit: bool = True
    compute_dtype: str = 'bf16'
    max_trans_aug: float = 0.125
    val_split: float = 0.1
    # Environment
    scene_bounds: tuple[float, ...] = (-0.7, -0.25, -0.1, -0.2, 0.25, 0.4)
    scene_bins: int = 32
    rot_bins: int = 72
    time_limit: int = 4
    num_demos_per_task: int = 100

    seed: int = 1
    datasets_dir: str = 'datasets/put_in_box'
    logdir: str = 'logdir/put_in_box_ff0'

    def save(self, file_path: str) -> None:
        """Save as YAML in a specified path."""
        yaml = YAML(typ="safe", pure=True)
        with open(file_path, "w", encoding="utf-8") as config_file:
            yaml.dump(dataclasses.asdict(self), config_file)

    @classmethod
    def load(cls, file_path: str, **kwargs) -> 'Config':
        """Load config from a YAML. Then values are updated by kwargs."""
        yaml = YAML(typ="safe", pure=True)
        with open(file_path, "r", encoding="utf-8") as config_file:
            config_dict = yaml.load(config_file)
        known_fields = map(lambda f: f.name, dataclasses.fields(cls))
        config_dict.update({k: v for k, v in kwargs.items()
                            if k in known_fields})
        def maybe_tuple(x): return tuple(x) if isinstance(x, list) else x
        return cls(**{k: maybe_tuple(v) for k, v in config_dict.items()})


peract_config = Config(
    conv_stem_features=(64, 64,),
    conv_stem_kernels=(1, 5),
    conv_stem_strides=(1, 5),
    conv_stem_use_skip_connections=True,
    latent_dim=2048,
    latent_channels=512,
    num_blocks=1,
    num_self_attend_per_block=6,
    num_cross_attend_heads=1,
    num_self_attend_heads=8,
    cross_attend_widening_factor=4.,
    self_attend_widening_factor=4.,
    use_trainable_pos_encoding=True,
    text_context_length=77,
    act_decoder_mlp_dim=256,
    act_decoder_conv_kernel=3,
    training_steps=600_000,
    batch_size=16,
    warmup_steps=-1,
    weight_decay=1e-6,
    peak_learning_rate=5e-4,
    rot_bins=72,
    scene_bins=100,
)

