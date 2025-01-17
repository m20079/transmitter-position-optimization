from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from jax import Array


@jax.tree_util.register_pytree_node_class
class PolynomialKernel(Kernel):
    def __init__(self: "PolynomialKernel", power: Array) -> None:
        self.power: Array = power

    def tree_flatten(self: Self) -> tuple[tuple[Array], None]:
        return (self.power,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children: tuple[Array]) -> "PolynomialKernel":
        return cls(*children)

    @partial(jax.jit, static_argnums=(0,))
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        input_abs: Array = jnp.abs(input1[0] - input2[0])
        return (
            input1[0] * input2[0] * parameter[0] + parameter[1]
        ) ** self.power + self.delta(input_abs) * parameter[2]

    @partial(jax.jit, static_argnums=(0,))
    def gradient(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])

    @partial(jax.jit, static_argnums=(0,))
    def hessian_matrix(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])


@jax.tree_util.register_pytree_node_class
class PolynomialTwoDimKernel(Kernel):
    def __init__(self: "PolynomialTwoDimKernel", power: Array) -> None:
        self.power: Array = power

    def tree_flatten(self: Self) -> tuple[tuple[Array], None]:
        return (self.power,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "PolynomialTwoDimKernel":
        return cls(*children)

    @partial(jax.jit, static_argnums=(0,))
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        input_abs: Array = jnp.sqrt(
            jnp.power(input1[0] - input2[0], 2) + jnp.power(input1[1] - input2[1], 2)
        )
        return (
            input1[0] * input2[0] * parameter[0]
            + input1[1] * input2[1] * parameter[1]
            + parameter[2]
        ) ** self.power + self.delta(input_abs) * parameter[3]

    @partial(jax.jit, static_argnums=(0,))
    def gradient(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])

    @partial(jax.jit, static_argnums=(0,))
    def hessian_matrix(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])
