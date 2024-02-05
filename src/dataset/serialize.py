from typing import Any

import numpy as np
from jax import tree_util

PyTree = Any


def serialize(obj: PyTree, path: str) -> None:
    leaves, tree = tree_util.tree_flatten(obj)
    np.savez_compressed(path, tree, *leaves)


def deserialize(path: str) -> PyTree:
    data = np.load(path, allow_pickle=True)
    tree, *data = list(data.values())
    def strip_obj(t): return t.item() if t.dtype.kind in 'OSU' else t
    data = tree_util.tree_map(strip_obj, data)
    return tree.item().unflatten(data)
