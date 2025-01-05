from functools import partial

import jax
from jax import Array


class Evaluation:
    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def min(data_rate: Array, axis: int) -> Array:
        return data_rate.min(axis=axis)

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def arithmetic_mean(data_rate: Array, axis: int) -> Array:
        return data_rate.mean(axis=axis)

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def geometric_mean(data_rate: Array, axis: int) -> Array:
        return data_rate.prod(axis=axis) ** (1.0 / data_rate.shape[-1])

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def standard_deviation(data_rate: Array, axis: int) -> Array:
        return data_rate.std(axis=axis)

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def range(data_rate: Array, axis: int) -> Array:
        return data_rate.max(axis=axis) - data_rate.min(axis=axis)
