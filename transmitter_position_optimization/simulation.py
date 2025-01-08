from typing import Literal

import jax.numpy as jnp
from bayesian_optimization.bayesian_optimization import (
    double_transmitter_optimization,
    single_transmitter_optimization,
    triple_transmitter_optimization,
)
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from environment.coordinate import Coordinate
from environment.propagation import Propagation
from environment.receivers import Receivers
from jax._src.pjit import JitWrapped


def single_transmitter_simulation(
    coordinate: Coordinate,
    propagation: Propagation,
    simulation_count: int,
    receiver_number: int,
    noise_floor: float,
    bandwidth: float,
    frequency: float,
    kernel: Kernel,
    parameter_optimization: ParameterOptimization,
    evaluation_function: JitWrapped,
    acquisition_function: JitWrapped,
    init_indices_pattern: Literal["random", "grid"],
    init_indices_number: int,
    debug_name: str,
) -> tuple[list[int], list[float], list[float]]:
    search_count: list[int] = []
    distance_error: list[float] = []
    data_rate_error: list[float] = []

    for seed in range(simulation_count):
        x_indices, y_indices = (
            coordinate.create_grid_transmitter_indices(number=init_indices_number)
            if init_indices_pattern == "grid"
            else coordinate.create_random_transmitter_indices(
                seed=seed, number=init_indices_number
            )
        )

        receivers: Receivers = coordinate.create_random_receivers(
            seed=seed,
            number=receiver_number,
            noise_floor=noise_floor,
            bandwidth=bandwidth,
        )

        result: tuple[int, float, float] = single_transmitter_optimization(
            propagation=propagation,
            receivers=receivers,
            coordinate=coordinate,
            frequency=frequency,
            init_x_position=float(coordinate.x_size / 2.0),
            init_y_position=float(coordinate.y_size / 2.0),
            kernel=kernel,
            parameter_optimization=parameter_optimization,
            evaluation_function=evaluation_function,
            acquisition_function=acquisition_function,
            x_train_indices=x_indices,
            y_train_indices=y_indices,
        )

        search_count.append(result[0])
        distance_error.append(result[1])
        data_rate_error.append(result[2])

        print(f"search_count: {result[0]}", flush=True)
        print(f"distance_error: {result[1]}", flush=True)
        print(f"data_rate_error: {result[2]}", flush=True)
        print(
            f"Simulation {debug_name} {seed + 1}/{simulation_count} completed.",
            flush=True,
        )

    return search_count, distance_error, data_rate_error


def double_transmitter_simulation(
    coordinate: Coordinate,
    propagation: Propagation,
    simulation_count: int,
    receiver_number: int,
    noise_floor: float,
    bandwidth: float,
    frequency: float,
    kernel: Kernel,
    parameter_optimization: ParameterOptimization,
    evaluation_function: JitWrapped,
    acquisition_function: JitWrapped,
    init_indices_pattern: Literal["random", "grid"],
    init_indices_number: int,
    debug_name: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    search_count: list[int] = []
    distance_error: list[float] = []
    each_distance_error: list[float] = []
    data_rate_error: list[float] = []

    for seed in range(simulation_count):
        x_indices_a, y_indices_a = (
            coordinate.create_grid_transmitter_indices(number=init_indices_number)
            if init_indices_pattern == "grid"
            else coordinate.create_random_transmitter_indices(
                seed=seed, number=init_indices_number
            )
        )
        x_indices_b, y_indices_b = (
            jnp.roll(
                coordinate.create_grid_transmitter_indices(number=init_indices_number),
                shift=1,
            )
            if init_indices_pattern == "grid"
            else coordinate.create_random_transmitter_indices(
                seed=seed + 1, number=init_indices_number
            )
        )

        receivers: Receivers = coordinate.create_random_receivers(
            seed=seed,
            number=receiver_number,
            noise_floor=noise_floor,
            bandwidth=bandwidth,
        )

        result: tuple[int, float, tuple[float, float], float] = (
            double_transmitter_optimization(
                propagation=propagation,
                receivers=receivers,
                coordinate=coordinate,
                frequency=frequency,
                init_x_position=float(coordinate.x_size / 2.0),
                init_y_position=float(coordinate.y_size / 2.0),
                kernel=kernel,
                parameter_optimization=parameter_optimization,
                evaluation_function=evaluation_function,
                acquisition_function=acquisition_function,
                x_train_indices_a=x_indices_a,
                y_train_indices_a=y_indices_a,
                x_train_indices_b=x_indices_b,
                y_train_indices_b=y_indices_b,
            )
        )

        search_count.append(result[0])
        distance_error.append(result[1])
        each_distance_error.append(result[2][0])
        each_distance_error.append(result[2][1])
        data_rate_error.append(result[3])

        print(f"search_count: {result[0]}", flush=True)
        print(f"distance_error: {result[1]}", flush=True)
        print(f"each_distance_error: {result[2]}", flush=True)
        print(f"data_rate_error: {result[3]}", flush=True)
        print(
            f"Simulation {debug_name} {seed + 1}/{simulation_count} completed.",
            flush=True,
        )

    return search_count, distance_error, each_distance_error, data_rate_error


def triple_transmitter_simulation(
    coordinate: Coordinate,
    propagation: Propagation,
    simulation_count: int,
    receiver_number: int,
    noise_floor: float,
    bandwidth: float,
    frequency: float,
    kernel: Kernel,
    parameter_optimization: ParameterOptimization,
    evaluation_function: JitWrapped,
    acquisition_function: JitWrapped,
    init_indices_pattern: Literal["random", "grid"],
    init_indices_number: int,
    debug_name: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    search_count: list[int] = []
    distance_error: list[float] = []
    each_distance_error: list[float] = []
    data_rate_error: list[float] = []

    for seed in range(simulation_count):
        x_indices_a, y_indices_a = (
            coordinate.create_grid_transmitter_indices(number=init_indices_number)
            if init_indices_pattern == "grid"
            else coordinate.create_random_transmitter_indices(
                seed=seed, number=init_indices_number
            )
        )
        x_indices_b, y_indices_b = (
            jnp.roll(
                coordinate.create_grid_transmitter_indices(number=init_indices_number),
                shift=1,
            )
            if init_indices_pattern == "grid"
            else coordinate.create_random_transmitter_indices(
                seed=seed + 1, number=init_indices_number
            )
        )
        x_indices_c, y_indices_c = (
            jnp.roll(
                coordinate.create_grid_transmitter_indices(number=init_indices_number),
                shift=2,
            )
            if init_indices_pattern == "grid"
            else coordinate.create_random_transmitter_indices(
                seed=seed + 2, number=init_indices_number
            )
        )

        receivers: Receivers = coordinate.create_random_receivers(
            seed=seed,
            number=receiver_number,
            noise_floor=noise_floor,
            bandwidth=bandwidth,
        )

        result: tuple[int, float, tuple[float, float, float], float] = (
            triple_transmitter_optimization(
                propagation=propagation,
                receivers=receivers,
                coordinate=coordinate,
                frequency=frequency,
                init_x_position=float(coordinate.x_size / 2.0),
                init_y_position=float(coordinate.y_size / 2.0),
                kernel=kernel,
                parameter_optimization=parameter_optimization,
                evaluation_function=evaluation_function,
                acquisition_function=acquisition_function,
                x_train_indices_a=x_indices_a,
                y_train_indices_a=y_indices_a,
                x_train_indices_b=x_indices_b,
                y_train_indices_b=y_indices_b,
                x_train_indices_c=x_indices_c,
                y_train_indices_c=y_indices_c,
            )
        )

        search_count.append(result[0])
        distance_error.append(result[1])
        each_distance_error.append(result[2][0])
        each_distance_error.append(result[2][1])
        each_distance_error.append(result[2][2])
        data_rate_error.append(result[3])

        print(f"search_count: {result[0]}", flush=True)
        print(f"distance_error: {result[1]}", flush=True)
        print(f"each_distance_error: {result[2]}", flush=True)
        print(f"data_rate_error: {result[3]}", flush=True)
        print(
            f"Simulation {debug_name} {seed + 1}/{simulation_count} completed.",
            flush=True,
        )

    return search_count, distance_error, each_distance_error, data_rate_error
