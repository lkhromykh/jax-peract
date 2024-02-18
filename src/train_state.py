from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import optax
from flax import core

Params: TypeAlias = jax.Array | core.FrozenDict[str, 'Params']


@jax.tree_util.register_pytree_node_class
class TrainState(NamedTuple):

    rng: jax.Array
    params: Params
    opt_state: optax.OptState
    step: jnp.int32

    tx: optax.TransformUpdateFn

    def update(self, *, grad: Params) -> 'TrainState':
        params = self.params
        opt_state = self.opt_state
        step = self.step
        updates, opt_state = self.tx(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return self._replace(
            params=params,
            opt_state=opt_state,
            step=optax.safe_int32_increment(step)
        )

    @classmethod
    def init(cls,
             *,
             rng: jax.Array,
             params: Params,
             optim: optax.GradientTransformation,
             ) -> 'TrainState':
        return cls(
            rng=rng,
            params=params,
            opt_state=optim.init(params),
            step=jnp.int32(0),
            tx=optim.update,
        )

    def replace(self, **kwargs) -> 'TrainState':
        return self._replace(**kwargs)

    def tree_flatten(self):
        children = (self.rng,
                    self.params,
                    self.opt_state,
                    self.step,
                    )
        aux = self.tx,
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)
