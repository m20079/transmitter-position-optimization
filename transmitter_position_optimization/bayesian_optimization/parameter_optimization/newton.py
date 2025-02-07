from functools import partial
from typing import Any, Self

import constant
import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from jax import Array


@jax.tree_util.register_pytree_node_class
class Newton(ParameterOptimization):
    def __init__(
        self: Self,
        count: int,
        parameter_optimization: ParameterOptimization,
    ) -> None:
        self.count: int = count
        self.parameter_optimization: ParameterOptimization = parameter_optimization

    def tree_flatten(self: Self) -> tuple[tuple[()], dict[str, Any]]:
        return (
            (),
            {
                "count": self.count,
                "parameter_optimization": self.parameter_optimization,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "Newton":
        return cls(*children, **aux_data)

    @partial(jax.jit, static_argnums=(3,))
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

        @jax.jit
        def body_fun(_, val: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            max_parameter, max_log_likelihood, parameter = val

            k: Array = kernel.create_k(
                input_train_data=input_train_data,
                parameter=parameter,
            )
            k_inv: Array = jnp.linalg.inv(k)
            gradient: Array = kernel.create_gradient(
                input_train_data=input_train_data,
                output_train_data=output_train_data,
                k_inv=k_inv,
                parameter=parameter,
            )
            hessian_matrix: Array = kernel.create_hessian_matrix(
                input_train_data=input_train_data,
                output_train_data=output_train_data,
                k_inv=k_inv,
                parameter=parameter,
            )
            hessian_matrix_inv: Array = jnp.linalg.inv(hessian_matrix)
            log_likelihood: Array = self.get_log_likelihood(
                k=k,
                k_inv=k_inv,
                output_train_data=output_train_data,
            )
            new_parameter: Array = jnp.clip(
                parameter - hessian_matrix_inv @ gradient, min=min, max=max
            )
            return (
                jnp.where(
                    log_likelihood > max_log_likelihood,
                    new_parameter,
                    max_parameter,
                ),
                jnp.where(
                    log_likelihood > max_log_likelihood,
                    log_likelihood,
                    max_log_likelihood,
                ),
                new_parameter,
            )

        init_val: tuple[Array, Array, Array] = (
            init_parameter,
            jnp.asarray(-jnp.inf, dtype=constant.floating),
            init_parameter,
        )

        parameter, _, _ = jax.lax.fori_loop(0, self.count, body_fun, init_val)

        return parameter
