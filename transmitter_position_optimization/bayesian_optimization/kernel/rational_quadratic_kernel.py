from functools import partial
from typing import Self

import constant
import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from jax import Array


class RationalQuadraticKernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range():
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @partial(jax.jit, static_argnums=(0,))
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


class RationalQuadraticTwoDimKernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range():
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

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
            parameter[0]
            * (
                1.0
                + jnp.power(input_abs, 2)
                / (2.0 * jnp.power(parameter[1], 2) * parameter[2])
            )
            ** -parameter[2]
            + self.delta(input_abs) * parameter[3]
        )

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


class RationalQuadraticPolynomialTwoDimKernel(Kernel):
    def __init__(self, power: int) -> None:
        self.power: int = power

    def tree_flatten(
        self: Self,
    ) -> tuple[tuple[()], dict[str, int]]:
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
            dtype=constant.floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

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


class RationalQuadraticPlusRationalQuadraticFourDimKernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @partial(jax.jit, static_argnums=(0,))
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


class RationalQuadraticTimesRationalQuadraticFourDimKernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @partial(jax.jit, static_argnums=(0,))
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


class RationalQuadraticPlusRationalQuadraticPlusRationalQuadraticSixDimKernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
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
            dtype=constant.floating,
        )

    @partial(jax.jit, static_argnums=(0,))
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


class RationalQuadraticTimesRationalQuadraticTimesRationalQuadraticSixDimKernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @partial(jax.jit, static_argnums=(0,))
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
