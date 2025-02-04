from typing import Callable

import jax
import jax.numpy as jnp
from constant import mean_max_delta, std_delta
from jax import Array
from jax._src.pjit import JitWrapped
from jax.scipy import stats


class Acquisition:
    @staticmethod
    def ucb(
        coefficient: Callable[[Array], Array] = lambda count: jnp.sqrt(
            jnp.log(count) / count
        ),
    ) -> JitWrapped:
        @jax.jit
        def function(
            mean: Array,
            std: Array,
            max: Array,
            count: Array,
        ) -> Array:
            return mean + coefficient(count) * std

        return function

    @staticmethod
    def pi() -> JitWrapped:
        @jax.jit
        def function(
            mean: Array,
            std: Array,
            max: Array,
            count: Array,
        ) -> Array:
            normal: Array = (mean - max - mean_max_delta) / (std + std_delta)
            return stats.norm.cdf(normal)

        return function

    @staticmethod
    def ei() -> JitWrapped:
        @jax.jit
        def function(
            mean: Array,
            std: Array,
            max: Array,
            count: Array,
        ) -> Array:
            normal: Array = (mean - max - mean_max_delta) / (std + std_delta)
            return (mean - max - mean_max_delta) * stats.norm.cdf(normal) + (
                std + std_delta
            ) * stats.norm.pdf(normal)

        return function
