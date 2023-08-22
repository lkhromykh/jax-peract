from typing import TypeVar

from jax import tree_util
import jax.numpy as jnp

T = TypeVar('T')


def tree_size(t: T) -> int:
    leaves = tree_util.tree_leaves(t)
    return sum(map(jnp.size, leaves))
