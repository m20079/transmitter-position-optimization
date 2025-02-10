from functools import partial

import jax
import jax.numpy as jnp
from constant import integer
from environment.coordinate import Coordinate
from environment.data_rate import DataRate
from environment.distance import get_distance
from environment.propagation import Propagation
from environment.receivers import Receivers
from jax import Array
from jax._src.pjit import JitWrapped


@partial(jax.jit, static_argnums=(0, 1, 4, 5, 6, 7))
def single_transmitter_distance_estimation(
    propagation: Propagation,
    coordinate: Coordinate,
    receivers_key: Array,
    shadowing_key: Array,
    receiver_number: int,
    noise_floor: float,
    bandwidth: float,
    evaluation_function: JitWrapped,
) -> tuple[int, float, float, float]:
    data_rate: DataRate = propagation.create_data_rate(
        coordinate=coordinate,
        receivers_key=receivers_key,
        shadowing_key=shadowing_key,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
    )
    transmitter_function: JitWrapped = data_rate.create_single_transmitter_function()

    receivers: Receivers = coordinate.create_random_position_receivers(
        key=receivers_key,
        number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
    )
    estimate_x_position: Array = jnp.average(receivers.x_positions)
    estimate_y_position: Array = jnp.average(receivers.y_positions)

    estimate_x_index, estimate_y_index = (
        coordinate.convert_transmitter_positions_to_indices(
            x_positions=estimate_x_position,
            y_positions=estimate_y_position,
        )
    )
    estimate_value: Array = evaluation_function(
        data_rate=transmitter_function(
            x_indices=estimate_x_index,
            y_indices=estimate_y_index,
        )
    )

    true_x_indices, true_y_indices = data_rate.get_true_single_transmitter_indices(
        evaluation_function=evaluation_function
    )
    true_x_positions, true_y_positions = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=true_x_indices,
            y_indices=true_y_indices,
        )
    )
    each_distances: Array = get_distance(
        x_positions_a=true_x_positions,
        y_positions_a=true_y_positions,
        x_positions_b=estimate_x_position,
        y_positions_b=estimate_y_position,
    )
    min_distance: Array = each_distances.min()

    true_data_rate: Array = evaluation_function(
        data_rate=transmitter_function(
            x_indices=true_x_indices,
            y_indices=true_y_indices,
        )
    )
    data_rate_absolute_error: Array = true_data_rate.max() - estimate_value.max()
    data_rate_relative_error: Array = data_rate_absolute_error / true_data_rate.max()
    return (
        1,
        float(min_distance),
        float(data_rate_absolute_error),
        float(data_rate_relative_error),
    )


def double_transmitter_distance(
    propagation: Propagation,
    coordinate: Coordinate,
    receivers: Receivers,
    frequency: float,
    init_x_position: float,
    init_y_position: float,
    evaluation_function: JitWrapped,
    number: int,
    key: Array,
) -> tuple[int, float, tuple[float, float], float]:
    data_rate: DataRate = propagation.create_data_rate(
        coordinate=coordinate,
        receivers=receivers,
        frequency=frequency,
        init_x_position=init_x_position,
        init_y_position=init_y_position,
    )
    transmitter_function: JitWrapped = data_rate.create_double_transmitter_function()
    min_distance = jnp.asarray(0, dtype=integer)
    min_distance_a = jnp.asarray(0, dtype=integer)
    min_distance_b = jnp.asarray(0, dtype=integer)
    data_rate_error = jnp.asarray(0, dtype=integer)
    a_key, b_key = jax.random.split(key, 2)
    x_indices_a, y_indices_a = coordinate.create_random_transmitter_indices(
        key=a_key,
        number=number,
    )
    x_indices_b, y_indices_b = coordinate.create_random_transmitter_indices(
        key=b_key,
        number=number,
    )
    each_data_rate: Array = transmitter_function(
        x_indices_a=x_indices_a,
        y_indices_a=y_indices_a,
        x_indices_b=x_indices_b,
        y_indices_b=y_indices_b,
    )
    output_train_data: Array = evaluation_function(
        data_rate=each_data_rate,
        axis=1,
    )
    final_train_data_index: tuple[Array, ...] = jnp.unravel_index(
        indices=jnp.argmax(output_train_data),
        shape=output_train_data.shape,
    )
    final_x_position_a, final_y_position_a = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices_a[final_train_data_index[0]],
            y_indices=y_indices_a[final_train_data_index[0]],
        )
    )
    final_x_position_b, final_y_position_b = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices_b[final_train_data_index[0]],
            y_indices=y_indices_b[final_train_data_index[0]],
        )
    )
    max_indices: Array = data_rate.get_true_double_transmitter_indices(
        evaluation_function=evaluation_function
    )
    pure_max_indices: Array = max_indices[jnp.all(max_indices != -1, axis=1)]
    max_x_positions_a, max_y_positions_a = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=pure_max_indices.T.at[0].get(),
            y_indices=pure_max_indices.T.at[1].get(),
        )
    )
    max_x_positions_b, max_y_positions_b = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=pure_max_indices.T.at[2].get(),
            y_indices=pure_max_indices.T.at[3].get(),
        )
    )
    each_distances_a: Array = get_distance(
        x_positions_a=max_x_positions_a,
        y_positions_a=max_y_positions_a,
        x_positions_b=final_x_position_a,
        y_positions_b=final_y_position_a,
    )
    each_distances_b: Array = get_distance(
        x_positions_a=max_x_positions_b,
        y_positions_a=max_y_positions_b,
        x_positions_b=final_x_position_b,
        y_positions_b=final_y_position_b,
    )
    distance: Array = jnp.sqrt(each_distances_a**2 + each_distances_b**2)
    min_distance_index: tuple[Array, ...] = jnp.unravel_index(
        indices=jnp.argmin(distance),
        shape=distance.shape,
    )
    min_distance: Array = distance.min()
    min_distance_a: Array = each_distances_a[min_distance_index]
    min_distance_b: Array = each_distances_b[min_distance_index]
    each_max_data_rate: Array = transmitter_function(
        x_indices_a=pure_max_indices.T.at[0].get(),
        y_indices_a=pure_max_indices.T.at[1].get(),
        x_indices_b=pure_max_indices.T.at[2].get(),
        y_indices_b=pure_max_indices.T.at[3].get(),
    )
    max_data_rate: Array = evaluation_function(
        data_rate=each_max_data_rate.at[0].get(), axis=0
    )
    data_rate_error: Array = max_data_rate - output_train_data.max()

    return (
        number,
        float(min_distance.block_until_ready()),
        (
            float(min_distance_a.block_until_ready()),
            float(min_distance_b.block_until_ready()),
        ),
        float(data_rate_error.block_until_ready()),
    )


def triple_transmitter_distance(
    propagation: Propagation,
    coordinate: Coordinate,
    receivers: Receivers,
    frequency: float,
    init_x_position: float,
    init_y_position: float,
    evaluation_function: JitWrapped,
    number: int,
    key: Array,
) -> tuple[int, float, tuple[float, float, float], float]:
    data_rate: DataRate = propagation.create_data_rate(
        coordinate=coordinate,
        receivers=receivers,
        frequency=frequency,
        init_x_position=init_x_position,
        init_y_position=init_y_position,
    )
    transmitter_function: JitWrapped = data_rate.create_triple_transmitter_function()
    min_distance = jnp.asarray(0, dtype=integer)
    min_distance_a = jnp.asarray(0, dtype=integer)
    min_distance_b = jnp.asarray(0, dtype=integer)
    min_distance_c = jnp.asarray(0, dtype=integer)
    data_rate_error = jnp.asarray(0, dtype=integer)

    a_key, b_key, c_key = jax.random.split(key, 3)

    x_indices_a, y_indices_a = coordinate.create_random_transmitter_indices(
        seed=a_key,
        number=number,
    )
    x_indices_b, y_indices_b = coordinate.create_random_transmitter_indices(
        seed=b_key,
        number=number,
    )
    x_indices_c, y_indices_c = coordinate.create_random_transmitter_indices(
        seed=c_key,
        number=number,
    )
    each_data_rate: Array = transmitter_function(
        x_indices_a=x_indices_a,
        y_indices_a=y_indices_a,
        x_indices_b=x_indices_b,
        y_indices_b=y_indices_b,
        x_indices_c=x_indices_c,
        y_indices_c=y_indices_c,
    )
    output_train_data: Array = evaluation_function(
        data_rate=each_data_rate,
        axis=1,
    )
    final_train_data_index: tuple[Array, ...] = jnp.unravel_index(
        indices=jnp.argmax(output_train_data),
        shape=output_train_data.shape,
    )
    final_x_position_a, final_y_position_a = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices_a[final_train_data_index[0]],
            y_indices=y_indices_a[final_train_data_index[0]],
        )
    )
    final_x_position_b, final_y_position_b = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices_b[final_train_data_index[0]],
            y_indices=y_indices_b[final_train_data_index[0]],
        )
    )
    final_x_position_c, final_y_position_c = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices_c[final_train_data_index[0]],
            y_indices=y_indices_c[final_train_data_index[0]],
        )
    )
    max_indices = data_rate.get_true_triple_transmitter_indices(
        evaluation_function=evaluation_function
    )
    pure_max_indices: Array = max_indices[jnp.all(max_indices != -1, axis=1)]

    max_x_positions_a, max_y_positions_a = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=pure_max_indices.T.at[0].get(),
            y_indices=pure_max_indices.T.at[1].get(),
        )
    )
    max_x_positions_b, max_y_positions_b = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=pure_max_indices.T.at[2].get(),
            y_indices=pure_max_indices.T.at[3].get(),
        )
    )
    max_x_positions_c, max_y_positions_c = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=pure_max_indices.T.at[4].get(),
            y_indices=pure_max_indices.T.at[5].get(),
        )
    )
    each_distances_a: Array = get_distance(
        x_positions_a=max_x_positions_a,
        y_positions_a=max_y_positions_a,
        x_positions_b=final_x_position_a,
        y_positions_b=final_y_position_a,
    )
    each_distances_b: Array = get_distance(
        x_positions_a=max_x_positions_b,
        y_positions_a=max_y_positions_b,
        x_positions_b=final_x_position_b,
        y_positions_b=final_y_position_b,
    )
    each_distances_c: Array = get_distance(
        x_positions_a=max_x_positions_c,
        y_positions_a=max_y_positions_c,
        x_positions_b=final_x_position_c,
        y_positions_b=final_y_position_c,
    )
    distance: Array = jnp.sqrt(
        each_distances_a**2 + each_distances_b**2 + each_distances_c**2
    )
    min_distance_index: tuple[Array, ...] = jnp.unravel_index(
        indices=jnp.argmin(distance),
        shape=distance.shape,
    )
    min_distance: Array = distance.min()
    min_distance_a: Array = each_distances_a[min_distance_index]
    min_distance_b: Array = each_distances_b[min_distance_index]
    min_distance_c: Array = each_distances_c[min_distance_index]

    each_max_data_rate: Array = transmitter_function(
        x_indices_a=pure_max_indices.T.at[0].get(),
        y_indices_a=pure_max_indices.T.at[1].get(),
        x_indices_b=pure_max_indices.T.at[2].get(),
        y_indices_b=pure_max_indices.T.at[3].get(),
        x_indices_c=pure_max_indices.T.at[4].get(),
        y_indices_c=pure_max_indices.T.at[5].get(),
    )
    max_data_rate: Array = evaluation_function(
        data_rate=each_max_data_rate.at[0].get(), axis=0
    )
    data_rate_error: Array = max_data_rate - output_train_data.max()

    return (
        number,
        float(min_distance.block_until_ready()),
        (
            float(min_distance_a.block_until_ready()),
            float(min_distance_b.block_until_ready()),
            float(min_distance_c.block_until_ready()),
        ),
        float(data_rate_error.block_until_ready()),
    )
