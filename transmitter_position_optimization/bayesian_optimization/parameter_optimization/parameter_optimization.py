from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from bayesian_optimization.kernel.kernel import Kernel
from jax import Array


class ParameterOptimization(metaclass=ABCMeta):
    @abstractmethod
    @partial(jax.jit, static_argnums=(3,))
    def optimize(
        self: Self,
        input_train_data: Array,
        output_train_data: Array,
        kernel: Kernel,
    ) -> Array:
        pass

    @jax.jit
    def get_log_likelihood(
        self: Self,
        k: Array,
        k_inv: Array,
        output_train_data: Array,
    ) -> Array:
        return (
            -jnp.log(jnp.linalg.det(k))
            - output_train_data.T @ k_inv @ output_train_data
        )
