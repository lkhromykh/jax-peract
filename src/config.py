import dataclasses
from typing import TypeAlias

from ruamel.yaml import YAML

Layers: TypeAlias = tuple[int, ...]


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Config:
    # IO processors
    scene_bins: int = 32
    rot_bins: int = 72
    conv_stem_features: Layers = ()
    conv_stem_kernels: Layers = ()
    conv_stem_strides: Layers = ()
    conv_stem_use_skip_connections: bool = True
    voxels_patch_size: int = 4
    text_context_length: int = 77  # max. 77
    tokens_dim: int = 64
    act_decoder_mlp_dim: int = 256
    act_decoder_conv_kernel: int = 3
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
    prior_initial_scale: float = 0.02
    ff_num_bands: int = 4
    # Training
    max_grad_norm: float = 10.
    warmup_steps: int = -1
    peak_learning_rate: float = 5e-4
    training_steps: int = 200_000
    batch_size: int = 16
    weight_decay: float = 1e-4
    log_every: int = 500
    save_every: int = 5000
    jit: bool = True
    compute_dtype: str = 'f32'
    max_trans_aug: float = 0.125
    val_split: float = 0.0
    # Environment
    scene_bounds: tuple[float, ...] = (-0.7, -0.25, -0.1, -0.2, 0.25, 0.4)
    time_limit: int = 8
    num_demos_per_task: int = 60
    # Experiment
    seed: int = 1
    datasets_dir: str = 'datasets/parsed_teleop'
    logdir: str = 'logdir/teleopv2.14'

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
    scene_bins=100,
    rot_bins=72,
    conv_stem_features=(64,),
    conv_stem_kernels=(1,),
    conv_stem_strides=(1,),
    voxels_patch_size=5,
    conv_stem_use_skip_connections=True,
    latent_dim=2048,
    latent_channels=512,
    num_blocks=1,
    num_self_attend_per_block=6,
    num_cross_attend_heads=1,
    num_self_attend_heads=8,
    cross_attend_widening_factor=4.,
    self_attend_widening_factor=4.,
    ff_num_bands=32,  # pos_enc size 3 * (2 * 32 + 1) approx. 192 as in the paper.
    text_context_length=77,
    act_decoder_mlp_dim=256,
    act_decoder_conv_kernel=3,
    training_steps=600_000,
    batch_size=16,
    warmup_steps=-1,
    weight_decay=1e-6,
    peak_learning_rate=5e-4,
    compute_dtype='f32',
    scene_bounds=(-0.3, -0.5, 0.6, 0.7, 0.5, 1.6)
)
