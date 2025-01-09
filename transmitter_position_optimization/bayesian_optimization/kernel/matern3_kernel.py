from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from jax import Array


class Matern3Kernel(Kernel):
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
            * (1.0 + jnp.sqrt(3.0) * input_abs / parameter[1])
            * jnp.exp(-jnp.sqrt(3.0) * input_abs / parameter[1])
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


class Matern3TwoDimKernel(Kernel):
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
            * (1.0 + jnp.sqrt(3.0) * input_abs / parameter[1])
            * jnp.exp(-jnp.sqrt(3.0) * input_abs / parameter[1])
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

class DoubleMatern3TwoDimKernel(Kernel):
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
            * (1.0 + jnp.sqrt(3.0) * input_abs1 / parameter[1])
            * jnp.exp(-jnp.sqrt(3.0) * input_abs1 / parameter[1])
            + parameter[2]
            * (1.0 + jnp.sqrt(3.0) * input_abs2 / parameter[3])
            * jnp.exp(-jnp.sqrt(3.0) * input_abs2 / parameter[3])
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


class TripleMatern3TwoDimKernel(Kernel):
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
            * (1.0 + jnp.sqrt(3.0) * input_abs1 / parameter[1])
            * jnp.exp(-jnp.sqrt(3.0) * input_abs1 / parameter[1])
            + parameter[2]
            * (1.0 + jnp.sqrt(3.0) * input_abs2 / parameter[3])
            * jnp.exp(-jnp.sqrt(3.0) * input_abs2 / parameter[3])
            + parameter[4]
            * (1.0 + jnp.sqrt(3.0) * input_abs3 / parameter[5])
            * jnp.exp(-jnp.sqrt(3.0) * input_abs3 / parameter[5])
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
