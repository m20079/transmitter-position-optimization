import constant
import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_pytree_node_class
class Receivers:
    def __init__(
        self,
        x_positions: list[float] | Array,
        y_positions: list[float] | Array,
        noise_floor: list[float] | Array,
        bandwidth: list[float] | Array,
    ) -> None:
        self.x_positions: Array = jnp.asarray(x_positions, dtype=constant.floating)
        self.y_positions: Array = jnp.asarray(y_positions, dtype=constant.floating)
        self.noise_floor: Array = jnp.asarray(noise_floor, dtype=constant.floating)
        self.bandwidth: Array = jnp.asarray(bandwidth, dtype=constant.floating)

    def tree_flatten(self) -> tuple[tuple[Array, Array, Array, Array], None]:
        return (
            (
                self.x_positions,
                self.y_positions,
                self.noise_floor,
                self.bandwidth,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "Receivers":
        return cls(*children)
