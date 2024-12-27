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
class RandomSearch(ParameterOptimization):
    def __init__(
        self,
        count: int,
        seed: int,
        min_params: Array,  # TODO lower_bound
        max_params: Array,  # TODO upper_bound
    ) -> None:
        self.count: int = count
        self.seed: int = seed
        self.min_params: Array = min_params
        self.max_params: Array = max_params

    def tree_flatten(
        self,
    ) -> tuple[tuple[int, int, Array, Array], None]:
        return (
            (
                self.count,
                self.seed,
                self.min_params,
                self.max_params,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "ParameterOptimization":
        return cls(*children)

    @partial(jax.jit, static_argnums=(0, 3))
    def optimize(
        self: Self,
        input_train_data: Array,
        output_train_data: Array,
        kernel: Kernel,
    ) -> Array:
        keys: Array = random.split(random.key(self.seed), self.min_params.size)

        parameter: Array = jax.vmap(
            lambda key, min_param, max_param: random.uniform(
                key,
                (self.count,),
                minval=min_param,
                maxval=max_param,
            )
        )(keys, self.min_params, self.max_params)

        @jax.jit
        def body_fn(
            parameter_i: Array,
        ) -> Array:
            k: Array = kernel.create_k(input_train_data, parameter_i)
            k_inv: Array = jnp.linalg.inv(k)
            likelihood: Array = self.get_log_likelihood(k, k_inv, output_train_data)
            return jnp.where(likelihood < 0.0, likelihood, -jnp.inf)

        return parameter.T[jnp.argmax(jax.vmap(body_fn)(parameter.T))]
