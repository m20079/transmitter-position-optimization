from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from constant import max_log_likelihood
from jax import Array


@jax.tree_util.register_pytree_node_class
class GridSearch(ParameterOptimization):
    def __init__(
        self: Self,
        search_pattern: list[Array],
    ) -> None:
        self.search_pattern: list[Array] = search_pattern

    def tree_flatten(self: Self) -> tuple[tuple[()], dict[str, list[Array]]]:
        return (
            (),
            {
                "search_pattern": self.search_pattern,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "GridSearch":
        return cls(*children, **aux_data)

    @partial(jax.jit, static_argnums=(3,))
    def optimize(
        self: Self,
        input_train_data: Array,
        output_train_data: Array,
        kernel: Kernel,
    ) -> Array:
        parameter: Array = jnp.asarray(jnp.meshgrid(*(self.search_pattern))).reshape(
            len(self.search_pattern), -1
        )

        @jax.jit
        def body_fn(
            parameter_i: Array,
        ) -> Array:
            k: Array = kernel.create_k(
                input_train_data=input_train_data,
                parameter=parameter_i,
            )
            k_inv: Array = jnp.linalg.inv(k)
            likelihood: Array = self.get_log_likelihood(
                k=k,
                k_inv=k_inv,
                output_train_data=output_train_data,
            )
            return jnp.where(likelihood < max_log_likelihood, likelihood, -jnp.inf)

        return parameter.T[jnp.argmax(jax.vmap(body_fn)(parameter.T))]
