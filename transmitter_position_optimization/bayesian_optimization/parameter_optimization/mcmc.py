from functools import partial
from typing import Self

import constant
import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from jax import Array, random


@jax.tree_util.register_pytree_node_class
class MCMC(ParameterOptimization):
    def __init__(
        self,
        count: int,
        seed: int,
        sigma_params: Array,
        parameter_optimization: ParameterOptimization,
    ) -> None:
        self.count: int = count
        self.seed: int = seed
        self.sigma_params: Array = sigma_params
        self.parameter_optimization: ParameterOptimization = parameter_optimization

    def tree_flatten(
        self: Self,
    ) -> tuple[tuple[int, int, Array, ParameterOptimization], None]:
        return (
            (
                self.count,
                self.seed,
                self.sigma_params,
                self.parameter_optimization,
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
        min: Array = jnp.finfo(constant.floating).tiny.astype(constant.floating)
        max: Array = jnp.finfo(constant.floating).max.astype(constant.floating)

        init_parameter: Array = self.parameter_optimization.optimize(
            input_train_data, output_train_data, kernel
        )
        keys: Array = random.split(random.key(self.seed), self.sigma_params.size)
        parameter: Array = jax.vmap(
            lambda key, sigma_param: random.normal(
                key,
                (self.count,),
            )
            * sigma_param
        )(keys, self.sigma_params)

        @jax.jit
        def body_fun(
            val: tuple[Array, Array, Array, Array, Array],
            parameter_i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
            index, old_parameter, old_likelihood, max_parameter, max_likelihood = val
            new_parameter: Array = jnp.clip(old_parameter + parameter_i, min, max)
            k: Array = kernel.create_k(input_train_data, new_parameter)
            k_inv: Array = jnp.linalg.inv(k)
            likelihood: Array = self.get_log_likelihood(k, k_inv, output_train_data)
            accept_prob: Array = random.uniform(random.key(seed=index))
            return (
                (
                    index + 1,
                    jnp.where(
                        likelihood > old_likelihood,
                        new_parameter,
                        jnp.where(
                            (old_likelihood / likelihood) < accept_prob,
                            new_parameter,
                            old_parameter,
                        ),
                    ),
                    jnp.where(
                        likelihood > old_likelihood,
                        likelihood,
                        jnp.where(
                            (old_likelihood / likelihood) < accept_prob,
                            likelihood,
                            old_likelihood,
                        ),
                    ),
                    jnp.where(
                        jnp.logical_and(
                            likelihood > max_likelihood, likelihood < jnp.asarray(0.0)
                        ),
                        new_parameter,
                        max_parameter,
                    ),
                    jnp.where(
                        jnp.logical_and(
                            likelihood > max_likelihood, likelihood < jnp.asarray(0.0)
                        ),
                        likelihood,
                        max_likelihood,
                    ),
                ),
                None,
            )

        init_val: tuple[Array, Array, Array, Array, Array] = (
            jnp.asarray(0, dtype=constant.integer),
            init_parameter,
            jnp.asarray(-jnp.inf, dtype=constant.floating),
            init_parameter,
            jnp.asarray(-jnp.inf, dtype=constant.floating),
        )
        (_, _, _, max_parameter, _), _ = jax.lax.scan(body_fun, init_val, parameter.T)
        return max_parameter
