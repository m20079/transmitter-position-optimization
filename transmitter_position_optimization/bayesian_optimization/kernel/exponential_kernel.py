from functools import partial
from typing import Self

import constant
import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from jax import Array


class ExponentialKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return (
            parameter[0] * jnp.exp(-(jnp.abs(input1[0] - input2[0])) / parameter[1])
            + self.delta(input1[0] - input2[0]) * parameter[2]
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
        return jnp.asarray(
            [
                self.get_del_log_likelihood(
                    k_inv,
                    output_train_data,
                    self.del_k_del_parameter0(input1, input2, parameter),
                ),
                self.get_del_log_likelihood(
                    k_inv,
                    output_train_data,
                    self.del_k_del_parameter1(input1, input2, parameter),
                ),
                self.get_del_log_likelihood(
                    k_inv,
                    output_train_data,
                    self.del_k_del_parameter2(input1, input2, parameter),
                ),
            ],
            dtype=constant.floating,
        )

    @partial(jax.jit, static_argnums=(0,))
    def del_k_del_parameter0(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return parameter[0] * jnp.exp(-jnp.abs(input1[0] - input2[0]) / parameter[1])

    @partial(jax.jit, static_argnums=(0,))
    def del_k_del_parameter1(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return self.del_k_del_parameter0(input1, input2, parameter) * (
            jnp.abs(input1[0] - input2[0]) / parameter[1]
        )

    @partial(jax.jit, static_argnums=(0,))
    def del_k_del_parameter2(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return self.delta(input1[0] - input2[0]) * parameter[2]

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


class ExponentialTwoDimKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return (
            parameter[0]
            * jnp.exp(
                -jnp.sqrt(
                    jnp.power(input1[0] - input2[0], 2)
                    + jnp.power(input1[1] - input2[1], 2)
                )
                / parameter[1]
            )
            + self.delta(
                jnp.abs(input1[0] - input2[0]) + jnp.abs(input1[1] - input2[1])
            )
            * parameter[2]
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
        return jnp.asarray(
            [
                self.get_del_log_likelihood(
                    k_inv,
                    output_train_data,
                    self.del_k_del_parameter0(input1, input2, parameter),
                ),
                self.get_del_log_likelihood(
                    k_inv,
                    output_train_data,
                    self.del_k_del_parameter1(input1, input2, parameter),
                ),
                self.get_del_log_likelihood(
                    k_inv,
                    output_train_data,
                    self.del_k_del_parameter2(input1, input2, parameter),
                ),
            ],
            dtype=constant.floating,
        )

    @partial(jax.jit, static_argnums=(0,))
    def del_k_del_parameter0(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return parameter[0] * jnp.exp(
            -jnp.sqrt(
                (
                    jnp.power(input1[0] - input2[0], 2)
                    + jnp.power(input1[1] - input2[1], 2)
                )
            )
            / parameter[1]
        )

    @partial(jax.jit, static_argnums=(0,))
    def del_k_del_parameter1(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return self.del_k_del_parameter0(input1, input2, parameter) * (
            jnp.sqrt(
                (
                    jnp.power(input1[0] - input2[0], 2)
                    + jnp.power(input1[1] - input2[1], 2)
                )
            )
            / parameter[1]
        )

    @partial(jax.jit, static_argnums=(0,))
    def del_k_del_parameter2(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return (
            self.delta(jnp.abs(input1[0] - input2[0]) + jnp.abs(input1[1] - input2[1]))
            * parameter[2]
        )

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
class ExponentialPolynomialTwoDimKernel(Kernel):
    def __init__(
        self,
        power: int,
    ) -> None:
        self.power = power

    def tree_flatten(
        self: Self,
    ) -> tuple[tuple[int], None]:
        return (
            (self.power,),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "ExponentialPolynomialTwoDimKernel":
        return cls(*children)

    @partial(jax.jit, static_argnums=(0,))
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return (
            parameter[0]
            * jnp.exp(
                -jnp.sqrt(
                    jnp.power(input1[0] - input2[0], 2)
                    + jnp.power(input1[1] - input2[1], 2)
                )
                / parameter[1]
            )
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


class ExponentialRationalQuadraticTwoDimKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return (
            parameter[0]
            * jnp.exp(
                -jnp.sqrt(
                    jnp.power(input1[0] - input2[0], 2)
                    + jnp.power(input1[1] - input2[1], 2)
                )
                / parameter[1]
            )
            + (
                1.0
                + (
                    jnp.power(input1[0] - input2[0], 2)
                    + jnp.power(input1[1] - input2[1], 2)
                )
                / (2.0 * jnp.power(parameter[2], 2) * parameter[3])
            )
            ** -parameter[4]
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


class DoubleExponentialTwoDimKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return (
            parameter[0]
            * jnp.exp(
                -(
                    jnp.sqrt(
                        jnp.power(input1[0] - input2[0], 2)
                        + jnp.power(input1[1] - input2[1], 2)
                    )
                )
                / parameter[1]
            )
            + parameter[2]
            * jnp.exp(
                -(
                    jnp.sqrt(
                        jnp.power(input1[2] - input2[2], 2)
                        + jnp.power(input1[3] - input2[3], 2)
                    )
                )
                / parameter[3]
            )
            + self.delta(
                jnp.abs(input1[0] - input2[0])
                + jnp.abs(input1[1] - input2[1])
                + jnp.abs(input1[2] - input2[2])
                + jnp.abs(input1[3] - input2[3])
            )
            * parameter[4]
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


class DoubleTimesExponentialTwoDimKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return (
            parameter[0]
            * jnp.exp(
                -(
                    jnp.sqrt(
                        jnp.power(input1[0] - input2[0], 2)
                        + jnp.power(input1[1] - input2[1], 2)
                    )
                )
                / parameter[1]
            )
            * jnp.exp(
                -(
                    jnp.sqrt(
                        jnp.power(input1[2] - input2[2], 2)
                        + jnp.power(input1[3] - input2[3], 2)
                    )
                )
                / parameter[2]
            )
            + self.delta(
                jnp.abs(input1[0] - input2[0])
                + jnp.abs(input1[1] - input2[1])
                + jnp.abs(input1[2] - input2[2])
                + jnp.abs(input1[3] - input2[3])
            )
            * parameter[3]
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


class TripleExponentialTwoDimKernel(Kernel):
    @partial(jax.jit, static_argnums=(0,))
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        return (
            parameter[0]
            * jnp.exp(
                -(
                    jnp.sqrt(
                        jnp.power(input1[0] - input2[0], 2)
                        + jnp.power(input1[1] - input2[1], 2)
                    )
                )
                / parameter[1]
            )
            + parameter[2]
            * jnp.exp(
                -(
                    jnp.sqrt(
                        jnp.power(input1[2] - input2[2], 2)
                        + jnp.power(input1[3] - input2[3], 2)
                    )
                )
                / parameter[3]
            )
            + parameter[4]
            * jnp.exp(
                -(
                    jnp.sqrt(
                        jnp.power(input1[4] - input2[4], 2)
                        + jnp.power(input1[5] - input2[5], 2)
                    )
                )
                / parameter[5]
            )
            + self.delta(
                jnp.abs(input1[0] - input2[0])
                + jnp.abs(input1[1] - input2[1])
                + jnp.abs(input1[2] - input2[2])
                + jnp.abs(input1[3] - input2[3])
                + jnp.abs(input1[4] - input2[4])
                + jnp.abs(input1[5] - input2[5])
            )
            * parameter[6]
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
