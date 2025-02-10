import jax
import jax.numpy as jnp
from constant import floating, integer
from environment.coordinate import Coordinate
from simulations import single_simulation

if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    jax.config.update("jax_platforms", "cpu")
    jax.config.update(
        "jax_enable_x64",
        integer == jnp.int64 and floating == jnp.float64,
    )

    coordinate = Coordinate(
        x_size=1000.0,
        y_size=1000.0,
        x_mesh=10,
        y_mesh=100,
    )
    print(coordinate.create_grid_single_transmitter_indices(number=2).T[0])
