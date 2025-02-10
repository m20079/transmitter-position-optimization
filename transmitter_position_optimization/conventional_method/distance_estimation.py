from functools import partial

import jax
import jax.numpy as jnp
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
        min_distance,
        data_rate_absolute_error,
        data_rate_relative_error,
    )
