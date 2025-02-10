import jax
import jax.numpy as jnp
from constant import floating, integer
from simulations import (
    double_transmitter_simulations,
    single_transmitter_simulations,
    triple_transmitter_simulations,
)

if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    jax.config.update("jax_platforms", "cpu")
    jax.config.update(
        "jax_enable_x64",
        integer == jnp.int64 and floating == jnp.float64,
    )

    single_transmitter_simulations()
    double_transmitter_simulations()
    triple_transmitter_simulations()
