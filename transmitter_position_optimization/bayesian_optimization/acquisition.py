from typing import Callable

import jax
import jax.numpy as jnp
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
            normal: Array = (mean - max) / std
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
            normal: Array = (mean - max) / std
            return (mean - max) * stats.norm.cdf(normal) + std * stats.norm.pdf(normal)

        return function
