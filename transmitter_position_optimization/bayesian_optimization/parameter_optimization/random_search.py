from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from constant import max_log_likelihood
from jax import Array, random


@jax.tree_util.register_pytree_node_class
class RandomSearch(ParameterOptimization):
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
    def tree_unflatten(cls, aux_data, children) -> "RandomSearch":
        return cls(*children, **aux_data)

    @partial(jax.jit, static_argnums=(3,))
    def optimize(
        self: Self,
        input_train_data: Array,
        output_train_data: Array,
        kernel: Kernel,
    ) -> Array:
        keys: Array = random.split(random.key(self.seed), self.lower_bound.size)

        parameter: Array = jax.vmap(
            lambda key, min_param, max_param: random.uniform(
                key,
                (self.count,),
                minval=min_param,
                maxval=max_param,
            )
        )(keys, self.lower_bound, self.upper_bound)

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
            return jnp.where(
                jnp.logical_and(
                    likelihood <= max_log_likelihood, likelihood != jnp.nan
                ),
                likelihood,
                -jnp.inf,
            )

        return parameter.T[jnp.argmax(jax.vmap(body_fn)(parameter.T))]
