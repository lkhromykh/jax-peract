import numpy as np
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


class Blockwise(tfd.Blockwise):

    def mode(self, *args, **kwargs):
        modes = map(lambda x: x.mode(*args, **kwargs), self.distributions)
        modes = jnp.atleast_1d(*modes)
        return jnp.concatenate(modes, -1)

    def mode_distributions(self, *args, **kwargs):
        values = [dist.mode(*args, **kwargs) for dist in self.distributions]
        return self.distributions, values


class JointDistributionSequential(tfd.JointDistributionSequential):
    """Restricted set of joint distributions: every dist event shape should be at most 1-dim.

    `Mode` methods are not proper distribution mode -- they are required only for deterministic policy evaluation.
    """

    def log_prob_parts(self, value, *args, **kwargs):
        """Split array into parts before using tfd API."""
        shapes = self.event_shape
        sections = np.cumsum([np.prod(shape, dtype=int) for shape in shapes])
        value_shape = zip(jnp.split(value, sections, -1), shapes)
        values = [value.reshape(shape) for value, shape in value_shape]
        return super().log_prob_parts(*values, *args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return sum(self.log_prob_parts(*args, **kwargs))

    def mode_distributions(self, *args, **kwargs):
        """Alike .sample_distributions."""
        distributions, values = [], []
        for fn in self._dist_fn_wrapped:
            dist = fn(*values)
            distributions.append(dist)
            values.append(dist.mode(*args, **kwargs))
        return distributions, values

    def mode(self, *args, **kwargs):
        _, values = self.mode_distributions(*args, **kwargs)
        values = jnp.atleast_1d(*values)
        return jnp.concatenate(values)


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
