import pathlib
from collections.abc import Iterator, Generator

import tree
import numpy as np
import tensorflow as tf

import src.types_ as types
from src.logger import get_logger
from src.environment import gcenv
from src.utils import serialize, deserialize
from src.dataset.keyframes_extraction import KeyframesExtractor, extractor_factory


class DemosDataset:
    """
    Manage demos.

    Dataset structure:
    . dataset_dir/
    |--- 00001.npz
    |--- 00002.npz
    |--- ...
    |--- *.npz (file name doesn't matter.)
    """

    def __init__(self,
                 dataset_dir: str | pathlib.Path,
                 cast_to_f16: bool = True,
                 raise_on_read_exc: bool = True
                 ) -> None:
        path = pathlib.Path(dataset_dir).resolve()
        assert path.exists(), f'Dataset is not found: {path}'
        self.dataset_dir = path
        self.cast_to_f16 = cast_to_f16
        self.raise_on_read_exc = raise_on_read_exc
        self._len = len(list(iter(self)))

    def __iter__(self) -> Iterator[pathlib.Path]:
        """Path generator."""
        return iter(sorted(self.dataset_dir.rglob('*.npz')))

    def as_demo_generator(self) -> Generator[gcenv.Observation, None, None]:
        """Plain demo generator."""
        for path in self:
            try:
                demo = deserialize(path)
            except Exception as exc:
                get_logger().warning('Read demo error %s: %s', path, exc)
                if self.raise_on_read_exc:
                    raise exc
                else:
                    continue
            else:
                yield demo

    def as_tf_dataset(self, extract_fn: KeyframesExtractor = extractor_factory()) -> tf.data.Dataset:
        """TensorFlow trajectory dataset."""
        def nested_stack(ts): return tree.map_structure(lambda *xs: np.stack(xs), *ts)

        def as_trajectory_generator():
            invalid_eps = 0
            ep_len, kf_num = [], []
            for idx, (path, demo) in enumerate(zip(self, self.as_demo_generator())):
                try:
                    pairs, kfs = extract_fn(demo)
                    ep_len.append(len(demo))
                    kf_num.append(len(kfs))
                    get_logger().info('%d. %s; Keyframes time steps: %s', idx, demo[0].goal, kfs)
                except AssertionError as exc:
                    get_logger().warning('Skipping ill-formed demo %s: %s', path, exc)
                    invalid_eps += 1
                    continue
                else:
                    observations, actions = map(nested_stack, zip(*pairs))
                    yield types.Trajectory(observations=observations, actions=actions)
            get_logger().info(
                f'Dataset {self.dataset_dir}, {idx + 1 - invalid_eps} episodes, stats (min/avg/max):\n'
                f'\tEpisode length: {min(ep_len), np.mean(ep_len).round(2), max(ep_len)}\n'
                f'\tNum keyframes: {min(kf_num), np.mean(kf_num).round(2), max(kf_num)}'
            )

        def to_tf_specs(x):
            return tf.TensorSpec(shape=(None,) + x.shape[1:], dtype=tf.as_dtype(x.dtype))
        sample = next(as_trajectory_generator())
        output_signature = tree.map_structure(to_tf_specs, sample)
        return tf.data.Dataset.from_generator(
            generator=as_trajectory_generator,
            output_signature=output_signature
        )

    def __len__(self):
        return self._len

    def append(self, demo: gcenv.Demo) -> None:
        demo = tree.map_structure(np.asarray, demo)
        if self.cast_to_f16:
            def to_f16(x): return x.astype(np.float16) if x.dtype.kind == 'f' else x
            demo = tree.map_structure(to_f16, demo)
        serialize(demo, self.dataset_dir / f'{self._len:05d}')
        self._len += 1
