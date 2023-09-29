import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray


def scaled_dot_product(query: Array,
                       key: Array,
                       value: Array,
                       ) -> Array:
    attn = jnp.einsum('...qhd,...khd->...hqk', query, key)
    attn /= np.sqrt(query.shape[-1])
    attn = jax.nn.softmax(attn)
    return jnp.einsum('...hqk,...khd->...qhd', attn, value)
