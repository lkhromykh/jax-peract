import logging
import pickle
import pathlib
from collections.abc import Iterable, Generator

import tree
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from src.dataset.serialize import serialize, deserialize
from src.environment import gcenv
import src.types_ as types


class DemosDataset:
    """

    . dataset_root
    |--- specs.pkl
    |--- demos/
       |--- 00001.npz
       |--- 00002.npz
       |--- ...
    """

    SPECS = 'specs.pkl'
    DEMOS_DIR = 'demos/'

    def __init__(self,
                 dataset_dir: str,
                 ) -> None:
        self.dataset_dir = pathlib.Path(dataset_dir).absolute()
        self.rel_path(DemosDataset.DEMOS_DIR).mkdir(parents=True, exist_ok=True)
        if not self.rel_path(DemosDataset.SPECS).exists():
            exc = RuntimeError(f'No environment specs found in {self.dataset_dir}')
            logging.error(str(exc))
            raise exc

    def as_generator(self) -> Generator[gcenv.Observation, None, None]:
        for demo in self.rel_path(DemosDataset.DEMOS_DIR).iterdir():
            demo = deserialize(demo)
            demo = tree.map_structure(lambda *ts: np.stack(ts), *demo)
            yield demo

    def as_tfdataset(self) -> tf.data.Dataset:
        obs_spec, _ = self.get_specs()
        def to_tf_specs(x): return tf.TensorSpec(shape=(None,) + x.shape[1:], dtype=tf.as_dtype(x.dtype))
        output_signature = tree.map_structure(to_tf_specs, obs_spec)
        return tf.data.Dataset.from_generator(
            generator=self.as_generator,
            output_signature=output_signature
        )

    def append(self, demo: gcenv.Demo) -> None:
        path = self.rel_path(DemosDataset.DEMOS_DIR)
        idx = len(list(path.iterdir())) + 1
        def to_f16(x): return x.astype(np.float16) if x.dtype.kind == 'f' else x
        demo = tree.map_structure(to_f16, demo)
        serialize(demo, path / f'{idx:05d}')

    def extend(self, demos: Iterable[gcenv.Demo]) -> None:
        [self.append(demo) for demo in demos]

    @classmethod
    def create_from_env(cls, dataset_dir: str, env: gcenv.GoalConditionedEnv) -> 'DemosDataset':
        dataloader = cls(dataset_dir)
        env_specs = env.observation_spec(), env.action_spec()
        with dataloader.rel_path(DemosDataset.SPECS).open(mode='wb') as f:
            pickle.dump(env_specs, f)
        return dataloader

    def get_specs(self) -> types.EnvSpecs:
        with self.rel_path(DemosDataset.SPECS).open(mode='rb') as f:
            return pickle.load(f)

    def rel_path(self, *args: str) -> pathlib.Path:
        return self.dataset_dir.joinpath(*args)
