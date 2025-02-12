from typing import Literal

import jax.numpy as jnp
from bayesian_optimization.bayesian_optimization import (
    triple_transmitter_bayesian_optimization,
)
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from conventional_method.random_search import (
    triple_transmitter_random_search,
)
from environment.coordinate import Coordinate
from environment.propagation import Propagation
from jax import Array, random
from jax._src.pjit import JitWrapped
from print import print_result
from save import save_result


def triple_transmitter_bo_simulation(
    propagation: Propagation,
    coordinate: Coordinate,
    receiver_number: int,
    noise_floor: float,
    bandwidth: float,
    kernel: Kernel,
    parameter_optimization: ParameterOptimization,
    evaluation_function: JitWrapped,
    acquisition_function: JitWrapped,
    simulation_number: int,
    init_train_indices_type: Literal["random", "grid"],
    init_train_indices_random_number: int,
    init_train_indices_grid_number: int,
) -> None:
    count: list[int] = []
    distance_error: list[float] = []
    distance_error_a: list[float] = []
    distance_error_b: list[float] = []
    distance_error_c: list[float] = []
    data_rate_absolute_error: list[float] = []
    data_rate_relative_error: list[float] = []

    debug_name: str = f"{kernel.__class__.__name__}_{init_train_indices_type}_{evaluation_function.__qualname__}_{acquisition_function.__qualname__}"

    for seed in range(simulation_number):
        receivers_key, shadowing_key, train_indices_key = random.split(
            key=random.key(seed=seed), num=3
        )

        train_indices_key_a, train_indices_key_b, train_indices_key_c = random.split(
            key=train_indices_key, num=3
        )

        grid_indices = coordinate.create_grid_triple_transmitter_indices(
            number=init_train_indices_grid_number,
        )

        x_train_indices_a, y_train_indices_a = jnp.where(
            init_train_indices_type == "random",
            coordinate.create_random_transmitter_indices(
                key=train_indices_key_a,
                number=init_train_indices_random_number,
            ),
            jnp.asarray([grid_indices[0], grid_indices[1]]),
        )

        x_train_indices_b, y_train_indices_b = jnp.where(
            init_train_indices_type == "random",
            coordinate.create_random_transmitter_indices(
                key=train_indices_key_b,
                number=init_train_indices_random_number,
            ),
            jnp.asarray([grid_indices[2], grid_indices[3]]),
        )

        x_train_indices_c, y_train_indices_c = jnp.where(
            init_train_indices_type == "random",
            coordinate.create_random_transmitter_indices(
                key=train_indices_key_c,
                number=init_train_indices_random_number,
            ),
            jnp.asarray([grid_indices[4], grid_indices[5]]),
        )

        result: tuple[int, float, float, float, float, float, float] = (
            triple_transmitter_bayesian_optimization(
                propagation=propagation,
                coordinate=coordinate,
                receivers_key=receivers_key,
                shadowing_key=shadowing_key,
                receiver_number=receiver_number,
                noise_floor=noise_floor,
                bandwidth=bandwidth,
                x_train_indices_a=x_train_indices_a,
                y_train_indices_a=y_train_indices_a,
                x_train_indices_b=x_train_indices_b,
                y_train_indices_b=y_train_indices_b,
                x_train_indices_c=x_train_indices_c,
                y_train_indices_c=y_train_indices_c,
                kernel=kernel,
                parameter_optimization=parameter_optimization,
                evaluation_function=evaluation_function,
                acquisition_function=acquisition_function,
            )
        )

        count.append(result[0])
        distance_error_a.append(result[1])
        distance_error_b.append(result[2])
        distance_error_c.append(result[3])
        distance_error.append(result[4])
        data_rate_absolute_error.append(result[5])
        data_rate_relative_error.append(result[6])

        print(
            f"simulation {debug_name}: {seed + 1}",
            flush=True,
        )
        print(f"count: {result[0]}", flush=True)
        print(f"distance_error_a: {result[1]}", flush=True)
        print(f"distance_error_b: {result[2]}", flush=True)
        print(f"distance_error_c: {result[3]}", flush=True)
        print(f"distance_error: {result[4]}", flush=True)
        print(f"data_rate_absolute_error: {result[5]}", flush=True)
        print(f"data_rate_relative_error: {result[6]}", flush=True)

    print_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        distance_error_a=distance_error_a,
        distance_error_b=distance_error_b,
        distance_error_c=distance_error_c,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
    )
    save_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        distance_error_a=distance_error_a,
        distance_error_b=distance_error_b,
        distance_error_c=distance_error_c,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
    )


def triple_transmitter_rs_simulation(
    propagation: Propagation,
    coordinate: Coordinate,
    transmitter_number: int,
    receiver_number: int,
    noise_floor: float,
    bandwidth: float,
    evaluation_function: JitWrapped,
    simulation_number: int,
) -> None:
    count: list[int] = []
    distance_error: list[float] = []
    distance_error_a: list[float] = []
    distance_error_b: list[float] = []
    distance_error_c: list[float] = []
    data_rate_absolute_error: list[float] = []
    data_rate_relative_error: list[float] = []

    debug_name: str = (
        f"random_search_{transmitter_number}_{evaluation_function.__qualname__}"
    )

    for seed in range(simulation_number):
        receivers_key, shadowing_key, transmitter_key = random.split(
            key=random.key(seed=seed), num=3
        )

        result: tuple[int, Array, Array, Array, Array, Array, Array] = (
            triple_transmitter_random_search(
                propagation=propagation,
                coordinate=coordinate,
                receivers_key=receivers_key,
                shadowing_key=shadowing_key,
                transmitter_key=transmitter_key,
                transmitter_number=transmitter_number,
                receiver_number=receiver_number,
                noise_floor=noise_floor,
                bandwidth=bandwidth,
                evaluation_function=evaluation_function,
            )
        )

        count.append(int(result[0]))
        distance_error_a.append(float(result[1].block_until_ready()))
        distance_error_b.append(float(result[2].block_until_ready()))
        distance_error_c.append(float(result[3].block_until_ready()))
        distance_error.append(float(result[4].block_until_ready()))
        data_rate_absolute_error.append(float(result[5].block_until_ready()))
        data_rate_relative_error.append(float(result[6].block_until_ready()))

        print(
            f"simulation {debug_name}: {seed + 1}",
            flush=True,
        )
        print(f"count: {int(result[0])}", flush=True)
        print(f"distance_error_a: {float(result[1].block_until_ready())}", flush=True)
        print(f"distance_error_b: {float(result[2].block_until_ready())}", flush=True)
        print(f"distance_error_c: {float(result[3].block_until_ready())}", flush=True)
        print(f"distance_error: {float(result[4].block_until_ready())}", flush=True)
        print(
            f"data_rate_absolute_error: {float(result[5].block_until_ready())}",
            flush=True,
        )
        print(
            f"data_rate_relative_error: {float(result[6].block_until_ready())}",
            flush=True,
        )

    print_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        distance_error_a=distance_error_a,
        distance_error_b=distance_error_b,
        distance_error_c=distance_error_c,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
    )
    save_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        distance_error_a=distance_error_a,
        distance_error_b=distance_error_b,
        distance_error_c=distance_error_c,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
    )
