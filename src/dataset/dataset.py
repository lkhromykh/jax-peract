import pathlib
from collections.abc import Iterable, Iterator, Generator

import tree
import numpy as np
from ml_dtypes import bfloat16
import tensorflow as tf

import src.types_ as types
from src.logger import get_logger
from src.environment import gcenv
from src.utils import serialize, deserialize
from src.dataset.keyframes_extraction import KeyframesExtractor, extractor_factory


# TODO: specify rw exceptions.
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
                 cast_to_bf16: bool = True,
                 raise_on_read_exc: bool = True
                 ) -> None:
        path = pathlib.Path(dataset_dir).resolve()
        assert path.exists(), f'Dataset is not found: {path}'
        self.dataset_dir = path
        self.cast_to_bf16 = cast_to_bf16
        self.raise_on_read_exc = raise_on_read_exc
        self._len = len(list(iter(self)))

    def __iter__(self) -> Iterator[pathlib.Path]:
        """Path generator."""
        return iter(sorted(self.dataset_dir.glob('*.npz')))

    def as_demo_generator(self) -> Generator[gcenv.Observation, None, None]:
        """Plain demo generator."""
        for path in iter(self):
            try:
                demo = deserialize(path)
            except Exception as exc:
                get_logger().warning('Read demo error %s\ndemo path %s', exc, path)
                if self.raise_on_read_exc:
                    raise exc
                else:
                    continue
            else:
                yield demo

    def as_tf_dataset(self, extract_fn: KeyframesExtractor = extractor_factory()) -> tf.data.Dataset:
        """TensorFlow trajectory dataset."""
        def as_trajectory_generator():
            for demo in self.as_demo_generator():
                pairs, _ = extract_fn(demo)
                def nested_stack(ts): return tree.map_structure(lambda *xs: np.stack(xs), *ts)
                obs, act = map(nested_stack, zip(*pairs))
                yield types.Trajectory(observations=obs, actions=act)

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
        if self.cast_to_bf16:
            def to_bf16(x): return x.astype(bfloat16) if x.dtype.kind == 'f' else x
            demo = tree.map_structure(to_bf16, demo)
        serialize(demo, self.dataset_dir / f'{self._len:05d}')
        self._len += 1

    def extend(self, demos: Iterable[gcenv.Demo]) -> None:
        [self.append(demo) for demo in demos]
