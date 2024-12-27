from typing import Final

import jax.numpy as jnp

integer: Final = jnp.int32
floating: Final = jnp.float32

# 0: fori_loop →　vmap
# 1: vmap → vmap
# 2: fori_loop → fori_loop
data_rate_generation: Final[int] = 0

double_transmitter_search_size = 1000
double_transmitter_max_size = 1000
triple_transmitter_search_size = 10
triple_transmitter_max_size = 10
