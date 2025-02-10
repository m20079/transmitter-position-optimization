from typing import Self

import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from constant import floating
from jax import Array


@jax.tree_util.register_pytree_node_class
class GaussianProcessRegression:
    def __init__(
        self: Self,
        input_train_data: Array,
        output_train_data: Array,
        parameter: Array,
        kernel: Kernel,
    ) -> None:
        self.input_train_data: Array = input_train_data
        self.output_train_data: Array = output_train_data
        self.parameter: Array = parameter
        self.kernel: Kernel = kernel

    def tree_flatten(
        self: Self,
    ) -> tuple[tuple[Array, Array, Array], dict[str, Kernel]]:
        return (
            (
                self.input_train_data,
                self.output_train_data,
                self.parameter,
            ),
            {
                "kernel": self.kernel,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "GaussianProcessRegression":
        return cls(*children, **aux_data)

    @jax.jit
    def function(
        self: Self,
        input_test_data: Array,
    ) -> Array:
        k: Array = self.kernel.create_k(self.input_train_data, self.parameter)
        k_star: Array = self.kernel.create_k_star(
            input_test_data, self.input_train_data, self.parameter
        )
        k_star_star: Array = self.kernel.create_k_star_star(
            input_test_data, self.parameter
        )

        k_inv: Array = jnp.linalg.pinv(k)
        mean: Array = k_star.T @ k_inv @ self.output_train_data
        std: Array = jnp.nan_to_num(
            jnp.sqrt(jnp.diag((k_star_star - k_star.T @ k_inv @ k_star)))
        )

        return jnp.asarray([mean, std], dtype=floating)
