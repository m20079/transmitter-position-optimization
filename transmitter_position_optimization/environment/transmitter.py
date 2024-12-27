from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from constant import floating
from environment.distance import get_distance
from jax import Array
from scipy.constants import c


@jax.tree_util.register_pytree_node_class
class Transmitter:
    def __init__(
        self,
        x_position: Array | float,
        y_position: Array | float,
        init_x_position: Array | float,
        init_y_position: Array | float,
        frequency: Array | float,
    ) -> None:
        self.x_position: Array = jnp.asarray(x_position, dtype=floating)
        self.y_position: Array = jnp.asarray(y_position, dtype=floating)
        self.init_x_position: Array = jnp.asarray(init_x_position, dtype=floating)
        self.init_y_position: Array = jnp.asarray(init_y_position, dtype=floating)
        self.wavelength: Array = jnp.asarray(c / frequency, dtype=floating)

    def tree_flatten(self) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
        return (
            (
                self.x_position,
                self.y_position,
                self.init_x_position,
                self.init_y_position,
                self.wavelength,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "Transmitter":
        return cls(*children)

    @partial(jax.jit, static_argnums=(0,))
    def get_transmitter_distance(
        self: Self,
        x_positions: Array,
        y_positions: Array,
    ) -> Array:
        return get_distance(
            x_positions_a=x_positions,
            y_positions_a=y_positions,
            x_positions_b=self.x_position,
            y_positions_b=self.y_position,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_delta_tx(
        self: Self,
    ) -> Array:
        return get_distance(
            x_positions_a=self.x_position,
            y_positions_a=self.y_position,
            x_positions_b=self.init_x_position,
            y_positions_b=self.init_y_position,
        )
