from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from jax import Array, random


@jax.tree_util.register_pytree_node_class
class LogRandomSearch(ParameterOptimization):
    def __init__(
        self: Self,
        lower_bound: Array,
        upper_bound: Array,
        count: int,
        seed: int,
    ) -> None:
        self.lower_bound: Array = lower_bound
        self.upper_bound: Array = upper_bound
        self.count: int = count
        self.seed: int = seed

    def tree_flatten(self: Self) -> tuple[tuple[Array, Array], dict[str, int]]:
        return (
            (
                self.lower_bound,
                self.upper_bound,
            ),
            {
                "count": self.count,
                "seed": self.seed,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "LogRandomSearch":
        return cls(*children, **aux_data)

    @partial(jax.jit, static_argnums=(3,))
    def optimize(
        self: Self,
        input_train_data: Array,
        output_train_data: Array,
        kernel: Kernel,
    ) -> Array:
        key: Array = random.key(self.seed)

        random_uniform: Array = random.uniform(
            key,
            (self.lower_bound.size, self.count),
        )

        parameter: Array = (
            jnp.expand_dims(self.lower_bound, axis=1)
            * (jnp.expand_dims(self.upper_bound / self.lower_bound, axis=1))
            ** random_uniform
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
            return jnp.where(likelihood < 0.0, likelihood, -jnp.inf)

        return parameter.T[jnp.argmax(jax.vmap(body_fn)(parameter.T))]
