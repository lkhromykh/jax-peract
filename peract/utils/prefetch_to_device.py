from typing import Any, TypeAlias
import itertools
import collections.abc

import jax

PyTree: TypeAlias = Any


def prefetch_to_device(iterator: collections.abc.Iterator[PyTree],
                       size: int = 2,
                       device: jax.Device | None = None
                       ) -> collections.abc.Iterator[PyTree]:
    # flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html#flax.jax_utils.prefetch_to_device
    queue = collections.deque()

    def enqueue(n):
        for data in itertools.islice(iterator, n):
            queue.append(jax.device_put(data, device))

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)
