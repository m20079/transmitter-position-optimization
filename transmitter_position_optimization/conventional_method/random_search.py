from functools import partial
import jax.numpy as jnp
from constant import integer
from environment.coordinate import Coordinate
from environment.data_rate import DataRate
from environment.distance import get_distance
from environment.propagation import Propagation
from environment.receivers import Receivers
from jax import Array
from jax._src.pjit import JitWrapped
import jax


@partial(jax.jit, static_argnums=(0, 1, 5, 6, 7, 8, 9))
def single_transmitter_random_search(
    propagation: Propagation,
    coordinate: Coordinate,
    receivers_key: Array,
    shadowing_key: Array,
    transmitter_key: Array,
    transmitter_number: int,
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

    x_indices, y_indices = coordinate.create_random_transmitter_indices(
        key=transmitter_key,
        number=transmitter_number,
    )
    max_data_rate: Array = evaluation_function(
        data_rate=transmitter_function(
            x_indices=x_indices,
            y_indices=y_indices,
        ),
    )
    max_data_rate_index: tuple[Array, ...] = jnp.unravel_index(
        indices=jnp.argmax(max_data_rate),
        shape=max_data_rate.shape,
    )
    max_x_position, max_y_position = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices[max_data_rate_index[0]],
            y_indices=y_indices[max_data_rate_index[0]],
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
        x_positions_b=max_x_position,
        y_positions_b=max_y_position,
    )
    min_distance: Array = each_distances.min()

    true_data_rate: Array = evaluation_function(
        data_rate=transmitter_function(
            x_indices=true_x_indices,
            y_indices=true_y_indices,
        )
    )
    data_rate_absolute_error: Array = true_data_rate.max() - max_data_rate.max()
    data_rate_relative_error: Array = data_rate_absolute_error / true_data_rate.max()
    return (
        transmitter_number,
        float(min_distance),
        float(data_rate_absolute_error),
        float(data_rate_relative_error),
    )


def double_transmitter_random_search(
    propagation: Propagation,
    coordinate: Coordinate,
    receivers: Receivers,
    frequency: float,
    init_x_position: float,
    init_y_position: float,
    evaluation_function: JitWrapped,
    seed: int,
    number: int,
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
    x_indices_a, y_indices_a = coordinate.create_random_transmitter_indices(
        seed=seed,
        number=number,
    )
    x_indices_b, y_indices_b = coordinate.create_random_transmitter_indices(
        seed=seed + 1,
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


def triple_transmitter_random_search(
    propagation: Propagation,
    coordinate: Coordinate,
    receivers: Receivers,
    frequency: float,
    init_x_position: float,
    init_y_position: float,
    evaluation_function: JitWrapped,
    seed: int,
    number: int,
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
    x_indices_a, y_indices_a = coordinate.create_random_transmitter_indices(
        seed=seed,
        number=number,
    )
    x_indices_b, y_indices_b = coordinate.create_random_transmitter_indices(
        seed=seed + 1,
        number=number,
    )
    x_indices_c, y_indices_c = coordinate.create_random_transmitter_indices(
        seed=seed + 2,
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
