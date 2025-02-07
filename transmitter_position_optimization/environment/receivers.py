from typing import Self

import jax
from jax import Array


@jax.tree_util.register_pytree_node_class
class Receivers:
    def __init__(
        self: Self,
        x_positions: Array,
        y_positions: Array,
        noise_floor: Array,
        bandwidth: Array,
    ) -> None:
        self.x_positions: Array = x_positions
        self.y_positions: Array = y_positions
        self.noise_floor: Array = noise_floor
        self.bandwidth: Array = bandwidth

    def tree_flatten(self: Self) -> tuple[tuple[Array, Array, Array, Array], dict]:
        return (
            (
                self.x_positions,
                self.y_positions,
                self.noise_floor,
                self.bandwidth,
            ),
            {},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "Receivers":
        return cls(*children, **aux_data)
