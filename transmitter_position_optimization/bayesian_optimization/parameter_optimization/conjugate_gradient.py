from functools import partial
from typing import Any, Self

import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from constant import floating
from jax import Array


@jax.tree_util.register_pytree_node_class
class ConjugateGradient(ParameterOptimization):
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
    def tree_unflatten(cls, aux_data, children) -> "ConjugateGradient":
        return cls(*children, **aux_data)

    @partial(jax.jit, static_argnums=(3,))
    def optimize(
        self: Self,
        input_train_data: Array,
        output_train_data: Array,
        kernel: Kernel,
    ) -> Array:
        min: Array = jnp.finfo(floating).tiny.astype(floating)
        max: Array = jnp.finfo(floating).max.astype(floating)

        init_parameter: Array = self.parameter_optimization.optimize(
            input_train_data=input_train_data,
            output_train_data=output_train_data,
            kernel=kernel,
        )

        init_k: Array = kernel.create_k(
            input_train_data=input_train_data,
            parameter=init_parameter,
        )
        init_k_inv: Array = jnp.linalg.inv(init_k)
        init_gradient: Array = kernel.create_gradient(
            input_train_data=input_train_data,
            output_train_data=output_train_data,
            k_inv=init_k_inv,
            parameter=init_parameter,
        )

        @jax.jit
        def body_fun(
            _, val: tuple[Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array]:
            max_parameter, max_log_likelihood, parameter, d = val

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

            g: Array = (
                -(
                    (jnp.expand_dims(gradient, axis=0) @ jnp.expand_dims(d, axis=1))
                    / (
                        jnp.expand_dims(d, axis=0)
                        @ hessian_matrix
                        @ jnp.expand_dims(d, axis=1)
                    )
                ).ravel()
                * d
            )

            next_parameter: Array = jnp.clip(parameter + g, min, max)
            next_k: Array = kernel.create_k(
                input_train_data=input_train_data,
                parameter=next_parameter,
            )
            next_k_inv: Array = jnp.linalg.inv(next_k)
            next_gradient: Array = kernel.create_gradient(
                input_train_data=input_train_data,
                output_train_data=output_train_data,
                k_inv=next_k_inv,
                parameter=next_parameter,
            )

            log_likelihood: Array = self.get_log_likelihood(
                k=next_k,
                k_inv=next_k_inv,
                output_train_data=output_train_data,
            )

            return (
                jnp.where(
                    log_likelihood > max_log_likelihood,
                    next_parameter,
                    max_parameter,
                ),
                jnp.where(
                    log_likelihood > max_log_likelihood,
                    log_likelihood,
                    max_log_likelihood,
                ),
                next_parameter,
                -next_gradient
                + (
                    (
                        (
                            jnp.expand_dims(next_gradient, axis=0)
                            @ hessian_matrix
                            @ jnp.expand_dims(d, axis=1)
                        )
                        / (
                            jnp.expand_dims(d, axis=0)
                            @ hessian_matrix
                            @ jnp.expand_dims(d, axis=1)
                        )
                    ).ravel()
                    * d
                ),
            )

        init_val: tuple[Array, Array, Array, Array] = (
            init_parameter,
            jnp.asarray(-jnp.inf, dtype=floating),
            init_parameter,
            -init_gradient,
        )

        parameter, _, _, _ = jax.lax.fori_loop(0, self.count, body_fun, init_val)

        return parameter
