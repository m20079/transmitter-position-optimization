import jax.numpy as jnp


def save_result(
    debug_name: str,
    count: list[int],
    distance_error: list[float],
    data_rate_absolute_error: list[float],
    data_rate_relative_error: list[float],
    each_distance_error_a: list[float] | None = None,
    each_distance_error_b: list[float] | None = None,
    each_distance_error_c: list[float] | None = None,
):
    jnp.savez(
        f"{debug_name}.npz",
        count=count,
        distance_error=distance_error,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
        each_distance_error_a=each_distance_error_a or jnp.array([]),
        each_distance_error_b=each_distance_error_b or jnp.array([]),
        each_distance_error_c=each_distance_error_c or jnp.array([]),
    )
