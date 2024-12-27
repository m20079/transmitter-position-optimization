import jax
from jax import Array


class Evaluation:
    @staticmethod
    @jax.jit
    def min(data_rate: Array) -> Array:
        return data_rate.min(axis=2)

    @staticmethod
    @jax.jit
    def arithmetic_mean(data_rate: Array) -> Array:
        return data_rate.mean(axis=2)

    @staticmethod
    @jax.jit
    def geometric_mean(data_rate: Array) -> Array:
        return data_rate.prod(axis=2) ** (1.0 / data_rate.shape[2])

    @staticmethod
    @jax.jit
    def standard_deviation(data_rate: Array) -> Array:
        return data_rate.std(axis=2)

    @staticmethod
    @jax.jit
    def range(data_rate: Array) -> Array:
        return data_rate.max(axis=2) - data_rate.min(axis=2)
