import jax
from jax import Array


class Evaluation:
    @staticmethod
    @jax.jit
    def min(data_rate: Array) -> Array:
        return data_rate.min(axis=data_rate.ndim - 1)

    @staticmethod
    @jax.jit
    def arithmetic_mean(data_rate: Array) -> Array:
        return data_rate.mean(axis=data_rate.ndim - 1)

    @staticmethod
    @jax.jit
    def geometric_mean(data_rate: Array) -> Array:
        return data_rate.prod(axis=data_rate.ndim - 1) ** (1.0 / data_rate.shape[-1])

    @staticmethod
    @jax.jit
    def standard_deviation(data_rate: Array) -> Array:
        return data_rate.std(axis=data_rate.ndim - 1)

    @staticmethod
    @jax.jit
    def range(data_rate: Array) -> Array:
        return data_rate.max(axis=data_rate.ndim - 1) - data_rate.min(
            axis=data_rate.ndim - 1
        )
