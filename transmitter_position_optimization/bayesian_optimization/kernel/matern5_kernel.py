from functools import partial
from typing import Self

import constant
import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from jax import Array


class Matern5Kernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e-1],
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
                + jnp.sqrt(5.0) * input_abs / parameter[1]
                + 5.0 / 3.0 * jnp.power(input_abs, 2) / jnp.power(parameter[1], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs / parameter[1])
            + self.delta(input_abs) * parameter[2]
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


class Matern5TwoDimKernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e-1],
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
                + jnp.sqrt(5.0) * input_abs / parameter[1]
                + 5.0 / 3.0 * jnp.power(input_abs, 2) / jnp.power(parameter[1], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs / parameter[1])
            + self.delta(input_abs) * parameter[2]
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


@jax.tree_util.register_pytree_node_class
class Matern5PolynomialTwoDimKernel(Kernel):
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
    def tree_unflatten(cls, aux_data, children) -> "Matern5PolynomialTwoDimKernel":
        return cls(*children, **aux_data)

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
        input_abs: Array = jnp.sqrt(
            jnp.power(input1[0] - input2[0], 2) + jnp.power(input1[1] - input2[1], 2)
        )
        return (
            parameter[0]
            * (
                1.0
                + jnp.sqrt(5.0) * input_abs / parameter[1]
                + 5.0 / 3.0 * jnp.power(input_abs, 2) / jnp.power(parameter[1], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs / parameter[1])
            + (
                input1[0] * input2[0] * parameter[2]
                + input1[1] * input2[1] * parameter[3]
                + parameter[4]
            )
            ** self.power
            + self.delta(
                jnp.abs(input1[0] - input2[0]) + jnp.abs(input1[1] - input2[1])
            )
            * parameter[5]
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


class Matern5PlusMatern5FourDimKernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
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
                + jnp.sqrt(5.0) * input_abs1 / parameter[1]
                + 5.0 / 3.0 * jnp.power(input_abs1, 2) / jnp.power(parameter[1], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs1 / parameter[1])
            + parameter[2]
            * (
                1.0
                + jnp.sqrt(5.0) * input_abs2 / parameter[3]
                + 5.0 / 3.0 * jnp.power(input_abs2, 2) / jnp.power(parameter[3], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs2 / parameter[3])
            + self.delta(input_abs1 + input_abs2) * parameter[4]
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


class Matern5TimesMatern5FourDimKernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
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
                + jnp.sqrt(5.0) * input_abs1 / parameter[1]
                + 5.0 / 3.0 * jnp.power(input_abs1, 2) / jnp.power(parameter[1], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs1 / parameter[1])
            * (
                1.0
                + jnp.sqrt(5.0) * input_abs2 / parameter[2]
                + 5.0 / 3.0 * jnp.power(input_abs2, 2) / jnp.power(parameter[2], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs2 / parameter[2])
            + self.delta(input_abs1 + input_abs2) * parameter[3]
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


class Matern5PlusMatern5PlusMatern5SixDimKernel(Kernel):
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
        input_abs3: Array = jnp.sqrt(
            jnp.power(input1[4] - input2[4], 2) + jnp.power(input1[5] - input2[5], 2)
        )
        return (
            parameter[0]
            * (
                1.0
                + jnp.sqrt(5.0) * input_abs1 / parameter[1]
                + 5.0 / 3.0 * jnp.power(input_abs1, 2) / jnp.power(parameter[1], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs1 / parameter[1])
            + parameter[2]
            * (
                1.0
                + jnp.sqrt(5.0) * input_abs2 / parameter[3]
                + 5.0 / 3.0 * jnp.power(input_abs2, 2) / jnp.power(parameter[3], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs2 / parameter[3])
            + parameter[4]
            * (
                1.0
                + jnp.sqrt(5.0) * input_abs3 / parameter[5]
                + 5.0 / 3.0 * jnp.power(input_abs3, 2) / jnp.power(parameter[5], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs3 / parameter[5])
            + self.delta(input_abs1 + input_abs2 + input_abs3) * parameter[6]
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


class Matern5TimesMatern5TimesMatern5SixDimKernel(Kernel):
    @staticmethod
    @jax.jit
    def random_search_range() -> Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=constant.floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> Array:
        return jnp.asarray(
            [
                [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-7],
                [1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e-1],
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
                + jnp.sqrt(5.0) * input_abs1 / parameter[1]
                + 5.0 / 3.0 * jnp.power(input_abs1, 2) / jnp.power(parameter[1], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs1 / parameter[1])
            * (
                1.0
                + jnp.sqrt(5.0) * input_abs2 / parameter[2]
                + 5.0 / 3.0 * jnp.power(input_abs2, 2) / jnp.power(parameter[2], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs2 / parameter[2])
            * (
                1.0
                + jnp.sqrt(5.0) * input_abs3 / parameter[3]
                + 5.0 / 3.0 * jnp.power(input_abs3, 2) / jnp.power(parameter[3], 2)
            )
            * jnp.exp(-jnp.sqrt(5.0) * input_abs3 / parameter[3])
            + self.delta(input_abs1 + input_abs2 + input_abs3) * parameter[4]
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
