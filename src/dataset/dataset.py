import pathlib
from collections.abc import Iterable, Generator

import tree
import numpy as np

from src.environment import gcenv
from src.utils import serialize, deserialize


# TODO: maybe supply with a keyframes_extractor and make visualisation/summary.
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
                 dataset_dir: str,
                 cast_to_f16: bool = True,
                 ) -> None:
        path = pathlib.Path(dataset_dir).absolute()
        assert path.exists(), f'Dataset is not found: {path}'
        self.dataset_dir = path
        self.cast_to_f16 = cast_to_f16

    def as_demo_generator(self) -> Generator[gcenv.Observation, None, None]:
        """Plain demo generator."""
        for path in self.dataset_dir.iterdir():
            demo = deserialize(path)
            yield demo

    def __len__(self):
        return len(list(self.dataset_dir.iterdir()))

    def append(self, demo: gcenv.Demo) -> None:
        idx = len(self) + 1
        demo = tree.map_structure(np.asanyarray, demo)
        if self.cast_to_f16:
            def to_f16(x): return x.astype(np.float16) if x.dtype.kind == 'f' else x
            demo = tree.map_structure(to_f16, demo)
        serialize(demo, self.dataset_dir / f'{idx:05d}')

    def extend(self, demos: Iterable[gcenv.Demo]) -> None:
        [self.append(demo) for demo in demos]
