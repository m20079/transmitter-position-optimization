import jax.numpy as jnp


def save_result(
    debug_name: str,
    count: list[int],
    distance_error: list[float],
    data_rate_absolute_error: list[float],
    data_rate_relative_error: list[float],
    distance_error_a: list[float] | None = None,
    distance_error_b: list[float] | None = None,
    distance_error_c: list[float] | None = None,
) -> None:
    jnp.savez(
        f"{debug_name}.npz",
        count=count,
        distance_error=distance_error,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
        distance_error_a=distance_error_a or jnp.array([]),
        distance_error_b=distance_error_b or jnp.array([]),
        distance_error_c=distance_error_c or jnp.array([]),
    )
