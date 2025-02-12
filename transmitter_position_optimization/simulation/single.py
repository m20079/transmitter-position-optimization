from typing import Literal

import jax.numpy as jnp
from bayesian_optimization.bayesian_optimization import (
    single_transmitter_bayesian_optimization,
)
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from conventional_method.distance_estimation import (
    single_transmitter_distance_estimation,
)
from conventional_method.random_search import (
    single_transmitter_random_search,
)
from environment.coordinate import Coordinate
from environment.propagation import Propagation
from jax import Array, random
from jax._src.pjit import JitWrapped
from print import print_result
from save import save_result


def single_transmitter_bo_simulation(
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
    data_rate_absolute_error: list[float] = []
    data_rate_relative_error: list[float] = []

    debug_name: str = f"{kernel.__class__.__name__}_{init_train_indices_type}_{evaluation_function.__qualname__}_{acquisition_function.__qualname__}"

    for seed in range(simulation_number):
        receivers_key, shadowing_key, train_indices_key = random.split(
            key=random.key(seed=seed), num=3
        )

        x_train_indices, y_train_indices = jnp.where(
            init_train_indices_type == "random",
            coordinate.create_random_transmitter_indices(
                key=train_indices_key,
                number=init_train_indices_random_number,
            ),
            coordinate.create_grid_single_transmitter_indices(
                number=init_train_indices_grid_number,
            ),
        )

        result: tuple[int, float, float, float] = (
            single_transmitter_bayesian_optimization(
                propagation=propagation,
                coordinate=coordinate,
                receivers_key=receivers_key,
                shadowing_key=shadowing_key,
                receiver_number=receiver_number,
                noise_floor=noise_floor,
                bandwidth=bandwidth,
                x_train_indices=x_train_indices,
                y_train_indices=y_train_indices,
                kernel=kernel,
                parameter_optimization=parameter_optimization,
                evaluation_function=evaluation_function,
                acquisition_function=acquisition_function,
            )
        )

        count.append(result[0])
        distance_error.append(result[1])
        data_rate_absolute_error.append(result[2])
        data_rate_relative_error.append(result[3])

        print(
            f"simulation {debug_name}: {seed + 1}",
            flush=True,
        )
        print(f"count: {result[0]}", flush=True)
        print(f"distance_error: {result[1]}", flush=True)
        print(f"data_rate_absolute_error: {result[2]}", flush=True)
        print(f"data_rate_relative_error: {result[3]}", flush=True)

    print_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
    )
    save_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
    )


def single_transmitter_de_simulation(
    propagation: Propagation,
    coordinate: Coordinate,
    receiver_number: int,
    noise_floor: float,
    bandwidth: float,
    evaluation_function: JitWrapped,
    simulation_number: int,
) -> None:
    count: list[int] = []
    distance_error: list[float] = []
    data_rate_absolute_error: list[float] = []
    data_rate_relative_error: list[float] = []

    debug_name: str = f"distance_estimation_{evaluation_function.__qualname__}"

    for seed in range(simulation_number):
        receivers_key, shadowing_key = random.split(key=random.key(seed=seed), num=2)

        result: tuple[int, Array, Array, Array] = (
            single_transmitter_distance_estimation(
                propagation=propagation,
                coordinate=coordinate,
                receivers_key=receivers_key,
                shadowing_key=shadowing_key,
                receiver_number=receiver_number,
                noise_floor=noise_floor,
                bandwidth=bandwidth,
                evaluation_function=evaluation_function,
            )
        )

        count.append(int(result[0]))
        distance_error.append(float(result[1].block_until_ready()))
        data_rate_absolute_error.append(float(result[2].block_until_ready()))
        data_rate_relative_error.append(float(result[3].block_until_ready()))

        print(
            f"simulation {debug_name}: {seed + 1}",
            flush=True,
        )
        print(f"count: {int(result[0])}", flush=True)
        print(f"distance_error: {float(result[1].block_until_ready())}", flush=True)
        print(
            f"data_rate_absolute_error: {float(result[2].block_until_ready())}",
            flush=True,
        )
        print(
            f"data_rate_relative_error: {float(result[3].block_until_ready())}",
            flush=True,
        )

    print_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
    )
    save_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
    )


def single_transmitter_rs_simulation(
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
    data_rate_absolute_error: list[float] = []
    data_rate_relative_error: list[float] = []

    debug_name: str = (
        f"random_search_{transmitter_number}_{evaluation_function.__qualname__}"
    )

    for seed in range(simulation_number):
        receivers_key, shadowing_key, transmitter_key = random.split(
            key=random.key(seed=seed), num=3
        )

        result: tuple[int, Array, Array, Array] = single_transmitter_random_search(
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

        count.append(int(result[0]))
        distance_error.append(float(result[1].block_until_ready()))
        data_rate_absolute_error.append(float(result[2].block_until_ready()))
        data_rate_relative_error.append(float(result[3].block_until_ready()))

        print(
            f"simulation {debug_name}: {seed + 1}",
            flush=True,
        )
        print(f"count: {int(result[0])}", flush=True)
        print(f"distance_error: {float(result[1].block_until_ready())}", flush=True)
        print(
            f"data_rate_absolute_error: {float(result[2].block_until_ready())}",
            flush=True,
        )
        print(
            f"data_rate_relative_error: {float(result[3].block_until_ready())}",
            flush=True,
        )

    print_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
    )
    save_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_absolute_error=data_rate_absolute_error,
        data_rate_relative_error=data_rate_relative_error,
    )
