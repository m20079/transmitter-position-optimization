from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Literal, Self

import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from constant import floating
from jax import Array
from jax._src.pjit import JitWrapped


@jax.tree_util.register_pytree_node_class
class ParameterOptimization(metaclass=ABCMeta):
    def tree_flatten(self: Self) -> tuple[tuple, dict]:
        return (
            (),
            {},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "ParameterOptimization":
        return cls(*children, **aux_data)

    @abstractmethod
    @partial(jax.jit, static_argnums=(3,))
    def optimize(
        self: Self,
        input_train_data: Array,
        output_train_data: Array,
        kernel: Kernel,
    ) -> Array:
        pass

    @staticmethod
    @jax.jit
    def get_log_likelihood(
        k: Array,
        k_inv: Array,
        output_train_data: Array,
    ) -> Array:
        return (
            -jnp.log(jnp.linalg.det(k))
            - output_train_data.T @ k_inv @ output_train_data
        )

    @staticmethod
    def backtracking(
        condition_type: Literal["armijo", "wolfe", "strong_wolfe"],
        count: int,
        c1: float = 1.0e-4,
        c2: float = 0.9,
        p: float = 0.9,
    ) -> JitWrapped:
        @jax.jit
        def armijo(
            next_log_likelihood: Array,
            log_likelihood: Array,
            next_gradient: Array,
            gradient: Array,
            update_vector: Array,
            learning_rate: Array,
        ) -> Array:
            return (next_log_likelihood) >= (
                log_likelihood
                + c1
                * learning_rate
                * jnp.expand_dims(gradient, axis=0)
                @ jnp.expand_dims(update_vector, axis=1)
            )

        @jax.jit
        def curvature(
            next_log_likelihood: Array,
            log_likelihood: Array,
            next_gradient: Array,
            gradient: Array,
            update_vector: Array,
            learning_rate: Array,
        ) -> Array:
            return (
                jnp.expand_dims(next_gradient, axis=0)
                @ jnp.expand_dims(update_vector, axis=1)
            ) <= (
                c2
                * jnp.expand_dims(gradient, axis=0)
                @ jnp.expand_dims(update_vector, axis=1)
            )

        @jax.jit
        def strong_curvature(
            next_log_likelihood: Array,
            log_likelihood: Array,
            next_gradient: Array,
            gradient: Array,
            update_vector: Array,
            learning_rate: Array,
        ) -> Array:
            return (
                jnp.abs(
                    jnp.expand_dims(next_gradient, axis=0)
                    @ jnp.expand_dims(update_vector, axis=1)
                )
            ) <= (
                c2
                * jnp.abs(
                    jnp.expand_dims(gradient, axis=0)
                    @ jnp.expand_dims(update_vector, axis=1)
                )
            )

        @jax.jit
        def wolfe(
            next_log_likelihood: Array,
            log_likelihood: Array,
            next_gradient: Array,
            gradient: Array,
            update_vector: Array,
            learning_rate: Array,
        ) -> Array:
            return jnp.logical_and(
                armijo(
                    next_log_likelihood=next_log_likelihood,
                    log_likelihood=log_likelihood,
                    next_gradient=next_gradient,
                    gradient=gradient,
                    update_vector=update_vector,
                    learning_rate=learning_rate,
                ),
                curvature(
                    next_log_likelihood=next_log_likelihood,
                    log_likelihood=log_likelihood,
                    next_gradient=next_gradient,
                    gradient=gradient,
                    update_vector=update_vector,
                    learning_rate=learning_rate,
                ),
            )

        @jax.jit
        def strong_wolfe(
            next_log_likelihood: Array,
            log_likelihood: Array,
            next_gradient: Array,
            gradient: Array,
            update_vector: Array,
            learning_rate: Array,
        ) -> Array:
            return jnp.logical_and(
                armijo(
                    next_log_likelihood=next_log_likelihood,
                    log_likelihood=log_likelihood,
                    next_gradient=next_gradient,
                    gradient=gradient,
                    update_vector=update_vector,
                    learning_rate=learning_rate,
                ),
                strong_curvature(
                    next_log_likelihood=next_log_likelihood,
                    log_likelihood=log_likelihood,
                    next_gradient=next_gradient,
                    gradient=gradient,
                    update_vector=update_vector,
                    learning_rate=learning_rate,
                ),
            )

        @partial(jax.jit, static_argnums=(6,))
        def function(
            input_train_data: Array,
            output_train_data: Array,
            parameter: Array,
            gradient: Array,
            update_vector: Array,
            log_likelihood: Array,
            kernel: Kernel,
        ) -> Array:
            @jax.jit
            def body_fun(learning_rate: Array) -> Array:
                next_k: Array = kernel.create_k(
                    input_train_data=input_train_data,
                    parameter=parameter + learning_rate * update_vector,
                )
                next_k_inv: Array = jnp.linalg.inv(next_k)
                next_gradient: Array = kernel.create_gradient(
                    input_train_data=input_train_data,
                    output_train_data=output_train_data,
                    k_inv=next_k_inv,
                    parameter=parameter + learning_rate * update_vector,
                )
                next_log_likelihood: Array = ParameterOptimization.get_log_likelihood(
                    k=next_k,
                    k_inv=next_k_inv,
                    output_train_data=output_train_data,
                )

                condition: Array = jnp.where(
                    condition_type == "strong_wolfe",
                    strong_wolfe(
                        next_log_likelihood=next_log_likelihood,
                        log_likelihood=log_likelihood,
                        next_gradient=next_gradient,
                        gradient=gradient,
                        update_vector=update_vector,
                        learning_rate=learning_rate,
                    ),
                    jnp.where(
                        condition_type == "wolfe",
                        wolfe(
                            next_log_likelihood=next_log_likelihood,
                            log_likelihood=log_likelihood,
                            next_gradient=next_gradient,
                            gradient=gradient,
                            update_vector=update_vector,
                            learning_rate=learning_rate,
                        ),
                        armijo(
                            next_log_likelihood=next_log_likelihood,
                            log_likelihood=log_likelihood,
                            next_gradient=next_gradient,
                            gradient=gradient,
                            update_vector=update_vector,
                            learning_rate=learning_rate,
                        ),
                    ),
                )

                return jnp.where(condition, learning_rate, jnp.asarray(0.0))

            return jax.vmap(body_fun)(p ** jnp.arange(count, dtype=floating)).max()

        return function
