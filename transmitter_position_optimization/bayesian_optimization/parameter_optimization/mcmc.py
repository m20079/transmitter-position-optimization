from functools import partial
from typing import Any, Self

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
        std_params: Array,
        parameter_optimization: ParameterOptimization,
    ) -> None:
        self.count: int = count
        self.seed: int = seed
        self.std_params: Array = std_params
        self.parameter_optimization: ParameterOptimization = parameter_optimization

    def tree_flatten(self: Self) -> tuple[tuple[Array], dict[str, Any]]:
        return (
            (self.std_params,),
            {
                "count": self.count,
                "seed": self.seed,
                "parameter_optimization": self.parameter_optimization,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "MCMC":
        return cls(*children, **aux_data)

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
            input_train_data=input_train_data,
            output_train_data=output_train_data,
            kernel=kernel,
        )

        keys: Array = random.split(random.key(self.seed), self.std_params.size)
        parameter: Array = jax.vmap(
            lambda key, std_param: std_param
            * random.normal(
                key,
                (self.count,),
            )
        )(keys, self.std_params)

        @jax.jit
        def body_fun(
            val: tuple[Array, Array, Array, Array, Array],
            parameter_i: Array,
        ) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
            seed, old_parameter, old_likelihood, max_parameter, max_likelihood = val
            new_parameter: Array = jnp.clip(old_parameter + parameter_i, min, max)
            k: Array = kernel.create_k(
                input_train_data=input_train_data,
                parameter=new_parameter,
            )
            k_inv: Array = jnp.linalg.inv(k)
            likelihood: Array = self.get_log_likelihood(
                k=k,
                k_inv=k_inv,
                output_train_data=output_train_data,
            )
            key: Array = random.key(seed)
            accept_prob: Array = random.uniform(key)
            return (
                (
                    seed + 1,
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
