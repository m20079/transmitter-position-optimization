from functools import partial

import jax
import jax.numpy as jnp
from environment.coordinate import Coordinate
from environment.data_rate import DataRate
from environment.distance import get_distance
from environment.propagation import Propagation
from jax import Array
from jax._src.pjit import JitWrapped


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
) -> tuple[int, Array, Array, Array]:
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
        min_distance,
        data_rate_absolute_error,
        data_rate_relative_error,
    )


@partial(jax.jit, static_argnums=(0, 1, 5, 6, 7, 8, 9))
def double_transmitter_random_search(
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
) -> tuple[int, Array, Array, Array, Array, Array]:
    data_rate: DataRate = propagation.create_data_rate(
        coordinate=coordinate,
        receivers_key=receivers_key,
        shadowing_key=shadowing_key,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
    )
    transmitter_function: JitWrapped = data_rate.create_double_transmitter_function()

    key_a, key_b = jax.random.split(key=transmitter_key, num=2)

    x_indices_a, y_indices_a = coordinate.create_random_transmitter_indices(
        key=key_a,
        number=transmitter_number,
    )
    x_indices_b, y_indices_b = coordinate.create_random_transmitter_indices(
        key=key_b,
        number=transmitter_number,
    )
    max_data_rate: Array = evaluation_function(
        data_rate=transmitter_function(
            x_indices_a=x_indices_a,
            y_indices_a=y_indices_a,
            x_indices_b=x_indices_b,
            y_indices_b=y_indices_b,
        ),
    )
    max_data_rate_index: tuple[Array, ...] = jnp.unravel_index(
        indices=jnp.argmax(max_data_rate),
        shape=max_data_rate.shape,
    )
    max_x_position_a, max_y_position_a = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices_a[max_data_rate_index[0]],
            y_indices=y_indices_a[max_data_rate_index[0]],
        )
    )
    max_x_position_b, max_y_position_b = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices_b[max_data_rate_index[0]],
            y_indices=y_indices_b[max_data_rate_index[0]],
        )
    )
    impure_true_indices: Array = data_rate.get_true_double_transmitter_indices(
        evaluation_function=evaluation_function
    )
    true_indices: Array = impure_true_indices[
        jnp.all(impure_true_indices != -1, axis=1)
    ]
    true_x_positions_a, true_y_positions_a = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=true_indices.T.at[0].get(),
            y_indices=true_indices.T.at[1].get(),
        )
    )
    true_x_positions_b, true_y_positions_b = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=true_indices.T.at[2].get(),
            y_indices=true_indices.T.at[3].get(),
        )
    )
    each_distance_a: Array = get_distance(
        x_positions_a=true_x_positions_a,
        y_positions_a=true_y_positions_a,
        x_positions_b=max_x_position_a,
        y_positions_b=max_y_position_a,
    )
    each_distance_b: Array = get_distance(
        x_positions_a=true_x_positions_b,
        y_positions_a=true_y_positions_b,
        x_positions_b=max_x_position_b,
        y_positions_b=max_y_position_b,
    )
    each_distance: Array = jnp.sqrt(each_distance_a**2.0 + each_distance_b**2.0)
    min_distance_index: tuple[Array, ...] = jnp.unravel_index(
        indices=jnp.argmin(each_distance),
        shape=each_distance.shape,
    )
    min_distance_a: Array = each_distance_a[min_distance_index[0]]
    min_distance_b: Array = each_distance_b[min_distance_index[0]]
    min_distance: Array = jnp.sqrt(min_distance_a**2.0 + min_distance_b**2.0)

    true_data_rate: Array = evaluation_function(
        data_rate=transmitter_function(
            x_indices_a=true_indices.T.at[0].get(),
            y_indices_a=true_indices.T.at[1].get(),
            x_indices_b=true_indices.T.at[2].get(),
            y_indices_b=true_indices.T.at[3].get(),
        )
    )
    data_rate_absolute_error: Array = true_data_rate.max() - max_data_rate.max()
    data_rate_relative_error: Array = data_rate_absolute_error / true_data_rate.max()
    return (
        transmitter_number,
        min_distance_a,
        min_distance_b,
        min_distance,
        data_rate_absolute_error,
        data_rate_relative_error,
    )


@partial(jax.jit, static_argnums=(0, 1, 5, 6, 7, 8, 9))
def triple_transmitter_random_search(
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
) -> tuple[int, Array, Array, Array, Array, Array, Array]:
    data_rate: DataRate = propagation.create_data_rate(
        coordinate=coordinate,
        receivers_key=receivers_key,
        shadowing_key=shadowing_key,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
    )
    transmitter_function: JitWrapped = data_rate.create_triple_transmitter_function()

    key_a, key_b, key_c = jax.random.split(key=transmitter_key, num=3)

    x_indices_a, y_indices_a = coordinate.create_random_transmitter_indices(
        key=key_a,
        number=transmitter_number,
    )
    x_indices_b, y_indices_b = coordinate.create_random_transmitter_indices(
        key=key_b,
        number=transmitter_number,
    )
    x_indices_c, y_indices_c = coordinate.create_random_transmitter_indices(
        key=key_c,
        number=transmitter_number,
    )
    max_data_rate: Array = evaluation_function(
        data_rate=transmitter_function(
            x_indices_a=x_indices_a,
            y_indices_a=y_indices_a,
            x_indices_b=x_indices_b,
            y_indices_b=y_indices_b,
            x_indices_c=x_indices_c,
            y_indices_c=y_indices_c,
        ),
    )
    max_data_rate_index: tuple[Array, ...] = jnp.unravel_index(
        indices=jnp.argmax(max_data_rate),
        shape=max_data_rate.shape,
    )
    max_x_position_a, max_y_position_a = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices_a[max_data_rate_index[0]],
            y_indices=y_indices_a[max_data_rate_index[0]],
        )
    )
    max_x_position_b, max_y_position_b = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices_b[max_data_rate_index[0]],
            y_indices=y_indices_b[max_data_rate_index[0]],
        )
    )
    max_x_position_c, max_y_position_c = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_indices_c[max_data_rate_index[0]],
            y_indices=y_indices_c[max_data_rate_index[0]],
        )
    )
    impure_true_indices: Array = data_rate.get_true_triple_transmitter_indices(
        evaluation_function=evaluation_function
    )
    true_indices: Array = impure_true_indices[
        jnp.all(impure_true_indices != -1, axis=1)
    ]
    true_x_positions_a, true_y_positions_a = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=true_indices.T.at[0].get(),
            y_indices=true_indices.T.at[1].get(),
        )
    )
    true_x_positions_b, true_y_positions_b = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=true_indices.T.at[2].get(),
            y_indices=true_indices.T.at[3].get(),
        )
    )
    true_x_positions_c, true_y_positions_c = (
        coordinate.convert_indices_to_transmitter_positions(
            x_indices=true_indices.T.at[4].get(),
            y_indices=true_indices.T.at[5].get(),
        )
    )
    each_distance_a: Array = get_distance(
        x_positions_a=true_x_positions_a,
        y_positions_a=true_y_positions_a,
        x_positions_b=max_x_position_a,
        y_positions_b=max_y_position_a,
    )
    each_distance_b: Array = get_distance(
        x_positions_a=true_x_positions_b,
        y_positions_a=true_y_positions_b,
        x_positions_b=max_x_position_b,
        y_positions_b=max_y_position_b,
    )
    each_distance_c: Array = get_distance(
        x_positions_a=true_x_positions_c,
        y_positions_a=true_y_positions_c,
        x_positions_b=max_x_position_c,
        y_positions_b=max_y_position_c,
    )
    each_distance: Array = jnp.sqrt(
        each_distance_a**2.0 + each_distance_b**2.0 + each_distance_c**2.0
    )
    min_distance_index: tuple[Array, ...] = jnp.unravel_index(
        indices=jnp.argmin(each_distance),
        shape=each_distance.shape,
    )
    min_distance_a: Array = each_distance_a[min_distance_index[0]]
    min_distance_b: Array = each_distance_b[min_distance_index[0]]
    min_distance_c: Array = each_distance_c[min_distance_index[0]]
    min_distance: Array = jnp.sqrt(
        min_distance_a**2.0 + min_distance_b**2.0 + min_distance_c**2.0
    )

    true_data_rate: Array = evaluation_function(
        data_rate=transmitter_function(
            x_indices_a=true_indices.T.at[0].get(),
            y_indices_a=true_indices.T.at[1].get(),
            x_indices_b=true_indices.T.at[2].get(),
            y_indices_b=true_indices.T.at[3].get(),
            x_indices_c=true_indices.T.at[4].get(),
            y_indices_c=true_indices.T.at[5].get(),
        )
    )
    data_rate_absolute_error: Array = true_data_rate.max() - max_data_rate.max()
    data_rate_relative_error: Array = data_rate_absolute_error / true_data_rate.max()
    return (
        transmitter_number,
        min_distance_a,
        min_distance_b,
        min_distance_c,
        min_distance,
        data_rate_absolute_error,
        data_rate_relative_error,
    )
