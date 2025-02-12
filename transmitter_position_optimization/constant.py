from typing import Final

import jax.numpy as jnp

platforms: Final[str] = "cuda"

integer: Final = jnp.int32
floating: Final = jnp.float32


# 0: fori_loop →　vmap
# 1: vmap → vmap
# 2: fori_loop → fori_loop
data_rate_generation: Final[int] = 0

double_transmitter_search_size: Final[int] = 41
double_transmitter_max_size: Final[int] = 41
triple_transmitter_search_size: Final[int] = 41**2
triple_transmitter_max_size: Final[int] = 41**2

# EI PI param
mean_max_delta: Final[float] = 1.0e-5
std_delta: Final[float] = 1.0e-5

# max log likelihood
max_log_likelihood: Final[float] = 0.0

# data rate unit: bps → Mbps, Gbps
data_rate_unit: Final[float] = 1.0e-9
