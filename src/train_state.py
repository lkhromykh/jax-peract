from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import core

Params = core.FrozenDict[str, Any]


@jax.tree_util.register_pytree_node_class
class TrainState(NamedTuple):

    rng: jax.random.PRNGKey
    params: Params
    target_params: Params
    opt_state: optax.OptState
    step: jnp.ndarray
    tx: optax.TransformUpdateFn
    target_update_var: float

    def update(self, *, grad: Params) -> 'TrainState':
        params = self.params
        target_params = self.target_params
        opt_state = self.opt_state
        step = self.step

        updates, opt_state = self.tx(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        if (tv := self.target_update_var) > 0:
            target_params = optax.incremental_update(params, target_params, tv)
        return self._replace(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            step=step + 1
        )

    @classmethod
    def init(cls,
             *,
             rng: jax.random.PRNGKey,
             params: Params,
             optim: optax.GradientTransformation,
             target_update_var: float
             ) -> 'TrainState':
        return cls(
            rng=rng,
            params=params,
            target_params=params,
            opt_state=optim.init(params),
            step=jnp.int32(0),
            tx=optim.update,
            target_update_var=target_update_var,
        )

    def replace(self, **kwargs):
        return self._replace(**kwargs)

    def tree_flatten(self):
        children = (self.rng,
                    self.params,
                    self.target_params,
                    self.opt_state,
                    self.step
                    )
        aux = (self.tx, self.target_update_var)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)
