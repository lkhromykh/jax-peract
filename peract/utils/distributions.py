import numpy as np
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


class Blockwise(tfd.Blockwise):

    def mode(self, *args, **kwargs):
        # assuming all distributions are independent.
        modes = map(lambda x: x.mode(*args, **kwargs), self.distributions)
        modes = jnp.atleast_1d(*modes)
        return jnp.concatenate(modes, -1)


class Idx2Grid(tfp.bijectors.AutoCompositeTensorBijector):
    """Ravel/unravel index functions wrapper."""

    def __init__(self, shape, validate_args=False, name='idx2grid'):
        super().__init__(validate_args=validate_args,
                         is_constant_jacobian=True,
                         forward_min_event_ndims=0,
                         inverse_min_event_ndims=1,
                         dtype=jnp.int32,
                         parameters=dict(locals()),
                         name=name)
        self.shape = shape

    @property
    def _is_permutation(self):
        return True

    def _forward(self, x):
        idxs = jnp.unravel_index(x, self.shape)
        return jnp.stack(idxs, -1)

    def _inverse(self, y):
        return jnp.ravel_multi_index(y, self.shape, mode='clip')

    def _inverse_log_det_jacobian(self, y):
        return jnp.zeros([], y.dtype)

    def _forward_event_shape_tensor(self, input_shape):
        shape = [input_shape, [len(self.shape)]]
        return np.concatenate(shape, dtype=input_shape.dtype)

    def _inverse_event_shape_tensor(self, output_shape):
        return output_shape[:-1]

    def _forward_event_shape(self, input_shape):
        return input_shape + (len(self.shape),)

    def _inverse_event_shape(self, output_shape):
        return output_shape[:-1]

    def _forward_dtype(self, input_dtype, **kwargs):
        return self.dtype

    def _inverse_dtype(self, output_dtype, **kwargs):
        return self.dtype
