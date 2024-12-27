import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def get_distance(
    x_positions_a: Array,
    y_positions_a: Array,
    x_positions_b: Array,
    y_positions_b: Array,
) -> Array:
    return jnp.sqrt(
        (x_positions_a - x_positions_b) ** 2.0 + (y_positions_a - y_positions_b) ** 2.0
    )
