from functools import partial
from typing import Self

import constant
import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from jax import Array


@jax.tree_util.register_pytree_node_class
class GaussianProcessRegression:
    def __init__(
        self: "GaussianProcessRegression",
        input_train_data: Array,
        output_train_data: Array,
        kernel: Kernel,
        parameter: Array,
    ) -> None:
        self.input_train_data: Array = input_train_data
        self.output_train_data: Array = output_train_data
        self.kernel: Kernel = kernel
        self.parameter: Array = parameter

    def tree_flatten(self: Self) -> tuple[tuple[Array, Array, Kernel, Array], None]:
        return (
            (
                self.input_train_data,
                self.output_train_data,
                self.kernel,
                self.parameter,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "GaussianProcessRegression":
        return cls(*children)

    @partial(jax.jit, static_argnums=(0, 2))
    def function(
        self: Self,
        input_test_data: Array,
        output_shape: tuple[int, ...],
    ) -> Array:
        k: Array = self.kernel.create_k(self.input_train_data, self.parameter)
        k_star: Array = self.kernel.create_k_star(
            input_test_data, self.input_train_data, self.parameter
        )
        k_star_star: Array = self.kernel.create_k_star_star(
            input_test_data, self.parameter
        )

        k_inv: Array = jnp.linalg.pinv(k)
        mean: Array = (k_star.T @ k_inv @ self.output_train_data).reshape(output_shape)
        std: Array = jnp.nan_to_num(
            jnp.sqrt(jnp.diag((k_star_star - k_star.T @ k_inv @ k_star))).reshape(
                output_shape
            )
        )

        return jnp.asarray([mean, std], dtype=constant.floating)
