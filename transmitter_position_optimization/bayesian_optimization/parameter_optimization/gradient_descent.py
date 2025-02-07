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
class GradientDescent(ParameterOptimization):
    def __init__(
        self: Self,
        count: int,
        learning_rate: int,
        parameter_optimization: ParameterOptimization,
    ) -> None:
        self.count: int = count
        self.learning_rate: int = learning_rate
        self.parameter_optimization: ParameterOptimization = parameter_optimization

    def tree_flatten(self: Self) -> tuple[tuple[()], dict[str, Any]]:
        return (
            (),
            {
                "count": self.count,
                "learning_rate": self.learning_rate,
                "parameter_optimization": self.parameter_optimization,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "GradientDescent":
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
        def body_fun(_, parameter: Array) -> Array:
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
            return jnp.clip(parameter + self.learning_rate * gradient, min=min, max=max)

        return jax.lax.fori_loop(0, self.count, body_fun, init_parameter)
