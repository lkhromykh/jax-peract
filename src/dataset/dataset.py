import pathlib
from collections.abc import Iterable, Generator

import tree
import numpy as np

from src.environment import gcenv
from src.utils import serialize, deserialize


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
        path.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = path
        self.cast_to_f16 = cast_to_f16
        self._len = len(list(iter(self)))

    def __iter__(self) -> Generator[pathlib.Path, None, None]:
        """Path generator."""
        return self.dataset_dir.glob('*.npz')

    def as_demo_generator(self) -> Generator[gcenv.Observation, None, None]:
        """Plain demo generator."""
        for path in iter(self):
            demo = deserialize(path)
            yield demo

    def __len__(self):
        return self._len

    def append(self, demo: gcenv.Demo) -> None:
        self._len += 1
        demo = tree.map_structure(np.asarray, demo)
        if self.cast_to_f16:
            def to_f16(x): return x.astype(np.float16) if x.dtype.kind == 'f' else x
            demo = tree.map_structure(to_f16, demo)
        serialize(demo, self.dataset_dir / f'{self._len:05d}')

    def extend(self, demos: Iterable[gcenv.Demo]) -> None:
        [self.append(demo) for demo in demos]
