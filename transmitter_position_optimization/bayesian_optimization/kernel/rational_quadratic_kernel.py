from typing import Self

import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from constant import floating
from jax import Array


@jax.tree_util.register_pytree_node_class
class RationalQuadraticKernel(Kernel):
    def tree_flatten(self: Self) -> tuple[tuple[()], dict]:
        return (
            (),
            {},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "RationalQuadraticKernel":
        return cls(*children, **aux_data)

    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @jax.jit
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        input_abs: Array = jnp.abs(input1[0] - input2[0])
        return (
            parameter[0]
            * (
                1.0
                + jnp.power(input_abs, 2)
                / (2.0 * jnp.power(parameter[1], 2) * parameter[2])
            )
            ** -parameter[2]
            + self.delta(input_abs) * parameter[3]
        )

    @jax.jit
    def gradient(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])

    @jax.jit
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
class RationalQuadraticTwoDimKernel(Kernel):
    def tree_flatten(self: Self) -> tuple[tuple[()], dict]:
        return (
            (),
            {},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "RationalQuadraticTwoDimKernel":
        return cls(*children, **aux_data)

    @staticmethod
    @jax.jit
    def random_search_range():
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @jax.jit
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
            parameter[0]
            * (
                1.0
                + jnp.power(input_abs, 2)
                / (2.0 * jnp.power(parameter[1], 2) * parameter[2])
            )
            ** -parameter[2]
            + self.delta(input_abs) * parameter[3]
        )

    @jax.jit
    def gradient(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])

    @jax.jit
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
class RationalQuadraticPolynomialTwoDimKernel(Kernel):
    def __init__(self: Self, power: int) -> None:
        self.power: int = power

    def tree_flatten(self: Self) -> tuple[tuple[()], dict[str, int]]:
        return (
            (),
            {
                "power": self.power,
            },
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data, children
    ) -> "RationalQuadraticPolynomialTwoDimKernel":
        return cls(*children, **aux_data)

    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @jax.jit
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
            parameter[0]
            * (
                1.0
                + jnp.power(input_abs, 2)
                / (2.0 * jnp.power(parameter[1], 2) * parameter[2])
            )
            ** -parameter[2]
            + (
                input1[0] * input2[0] * parameter[3]
                + input1[1] * input2[1] * parameter[4]
                + parameter[5]
            )
            ** self.power
            + self.delta(input_abs) * parameter[6]
        )

    @jax.jit
    def gradient(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])

    @jax.jit
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
class RationalQuadraticPlusRationalQuadraticFourDimKernel(Kernel):
    def tree_flatten(self: Self) -> tuple[tuple[()], dict]:
        return (
            (),
            {},
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data, children
    ) -> "RationalQuadraticPlusRationalQuadraticFourDimKernel":
        return cls(*children, **aux_data)

    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @jax.jit
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        input_abs1: Array = jnp.sqrt(
            jnp.power(input1[0] - input2[0], 2) + jnp.power(input1[1] - input2[1], 2)
        )
        input_abs2: Array = jnp.sqrt(
            jnp.power(input1[2] - input2[2], 2) + jnp.power(input1[3] - input2[3], 2)
        )
        return (
            parameter[0]
            * (
                1.0
                + jnp.power(input_abs1, 2)
                / (2.0 * jnp.power(parameter[1], 2) * parameter[2])
            )
            ** -parameter[2]
            + parameter[3]
            * (
                1.0
                + jnp.power(input_abs2, 2)
                / (2.0 * jnp.power(parameter[4], 2) * parameter[5])
            )
            ** -parameter[5]
            + self.delta(input_abs1 + input_abs2) * parameter[6]
        )

    @jax.jit
    def gradient(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])

    @jax.jit
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
class RationalQuadraticTimesRationalQuadraticFourDimKernel(Kernel):
    def tree_flatten(self: Self) -> tuple[tuple[()], dict]:
        return (
            (),
            {},
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data, children
    ) -> "RationalQuadraticTimesRationalQuadraticFourDimKernel":
        return cls(*children, **aux_data)

    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @jax.jit
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        input_abs1: Array = jnp.sqrt(
            jnp.power(input1[0] - input2[0], 2) + jnp.power(input1[1] - input2[1], 2)
        )
        input_abs2: Array = jnp.sqrt(
            jnp.power(input1[2] - input2[2], 2) + jnp.power(input1[3] - input2[3], 2)
        )
        return (
            parameter[0]
            * (
                1.0
                + jnp.power(input_abs1, 2)
                / (2.0 * jnp.power(parameter[1], 2) * parameter[2])
            )
            ** -parameter[2]
            * (
                1.0
                + jnp.power(input_abs2, 2)
                / (2.0 * jnp.power(parameter[3], 2) * parameter[4])
            )
            ** -parameter[4]
            + self.delta(input_abs1 + input_abs2) * parameter[5]
        )

    @jax.jit
    def gradient(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])

    @jax.jit
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
class RationalQuadraticPlusRationalQuadraticPlusRationalQuadraticSixDimKernel(Kernel):
    def tree_flatten(self: Self) -> tuple[tuple[()], dict]:
        return (
            (),
            {},
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data, children
    ) -> "RationalQuadraticPlusRationalQuadraticPlusRationalQuadraticSixDimKernel":
        return cls(*children, **aux_data)

    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [
                    1.0e-2,
                    1.0e-2,
                    1.0e-2,
                    1.0e-2,
                    1.0e-2,
                    1.0e-2,
                    1.0e-2,
                    1.0e-2,
                    1.0e-2,
                    1.0e-7,
                ],
                [
                    1.0e4,
                    1.0e4,
                    1.0e4,
                    1.0e4,
                    1.0e4,
                    1.0e4,
                    1.0e4,
                    1.0e4,
                    1.0e4,
                    1.0e-1,
                ],
            ],
            dtype=floating,
        )

    @jax.jit
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        input_abs1: Array = jnp.sqrt(
            jnp.power(input1[0] - input2[0], 2) + jnp.power(input1[1] - input2[1], 2)
        )
        input_abs2: Array = jnp.sqrt(
            jnp.power(input1[2] - input2[2], 2) + jnp.power(input1[3] - input2[3], 2)
        )
        input_abs3: Array = jnp.sqrt(
            jnp.power(input1[4] - input2[4], 2) + jnp.power(input1[5] - input2[5], 2)
        )
        return (
            parameter[0]
            * (
                1.0
                + jnp.power(input_abs1, 2)
                / (2.0 * jnp.power(parameter[1], 2) * parameter[2])
            )
            ** -parameter[2]
            + parameter[3]
            * (
                1.0
                + jnp.power(input_abs2, 2)
                / (2.0 * jnp.power(parameter[4], 2) * parameter[5])
            )
            ** -parameter[5]
            + parameter[6]
            * (
                1.0
                + jnp.power(input_abs3, 2)
                / (2.0 * jnp.power(parameter[7], 2) * parameter[8])
            )
            ** -parameter[8]
            + self.delta(input_abs1 + input_abs2) * parameter[9]
        )

    @jax.jit
    def gradient(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])

    @jax.jit
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
class RationalQuadraticTimesRationalQuadraticTimesRationalQuadraticSixDimKernel(Kernel):
    def tree_flatten(self: Self) -> tuple[tuple[()], dict]:
        return (
            (),
            {},
        )

    @classmethod
    def tree_unflatten(
        cls, aux_data, children
    ) -> "RationalQuadraticTimesRationalQuadraticTimesRationalQuadraticSixDimKernel":
        return cls(*children, **aux_data)

    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @jax.jit
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        input_abs1: Array = jnp.sqrt(
            jnp.power(input1[0] - input2[0], 2) + jnp.power(input1[1] - input2[1], 2)
        )
        input_abs2: Array = jnp.sqrt(
            jnp.power(input1[2] - input2[2], 2) + jnp.power(input1[3] - input2[3], 2)
        )
        input_abs3: Array = jnp.sqrt(
            jnp.power(input1[4] - input2[4], 2) + jnp.power(input1[5] - input2[5], 2)
        )
        return (
            parameter[0]
            * (
                1.0
                + jnp.power(input_abs1, 2)
                / (2.0 * jnp.power(parameter[1], 2) * parameter[2])
            )
            ** -parameter[2]
            * (
                1.0
                + jnp.power(input_abs2, 2)
                / (2.0 * jnp.power(parameter[3], 2) * parameter[4])
            )
            ** -parameter[4]
            * (
                1.0
                + jnp.power(input_abs3, 2)
                / (2.0 * jnp.power(parameter[5], 2) * parameter[6])
            )
            ** -parameter[6]
            + self.delta(input_abs1 + input_abs2) * parameter[7]
        )

    @jax.jit
    def gradient(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])

    @jax.jit
    def hessian_matrix(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return jnp.asarray([])
