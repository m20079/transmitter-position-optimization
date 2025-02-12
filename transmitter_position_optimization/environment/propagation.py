from functools import partial
from typing import Any, Self

import jax
import jax.numpy as jnp
from constant import data_rate_generation, data_rate_unit, floating
from environment.coordinate import Coordinate
from environment.data_rate import DataRate
from environment.distance import get_distance
from environment.receivers import Receivers
from jax import Array, random
from scipy.constants import c


@jax.tree_util.register_pytree_node_class
class Propagation:
    def __init__(
        self: Self,
        init_transmitter_x_position: Array,
        init_transmitter_y_position: Array,
        free_distance: float,
        propagation_coefficient: float,
        distance_correlation: float,
        standard_deviation: float,
        frequency: float,
    ) -> None:
        self.init_transmitter_x_position: Array = init_transmitter_x_position
        self.init_transmitter_y_position: Array = init_transmitter_y_position
        self.free_distance: float = free_distance
        self.propagation_coefficient: float = propagation_coefficient
        self.distance_correlation: float = distance_correlation
        self.standard_deviation: float = standard_deviation
        self.frequency: float = frequency

    def tree_flatten(self: Self) -> tuple[tuple[Array, Array], dict[str, Any]]:
        return (
            (
                self.init_transmitter_x_position,
                self.init_transmitter_y_position,
            ),
            {
                "free_distance": self.free_distance,
                "propagation_coefficient": self.propagation_coefficient,
                "distance_correlation": self.distance_correlation,
                "standard_deviation": self.standard_deviation,
                "frequency": self.frequency,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "Propagation":
        return cls(*children, **aux_data)

    @partial(jax.jit, static_argnums=(1,))
    def create_pathloss(
        self: Self,
        coordinate: Coordinate,
        transmitter_x_position: Array,
        transmitter_y_position: Array,
    ) -> Array:
        wavelength: float = c / self.frequency
        x_positions, y_positions = coordinate.create_all_receiver_positions()
        distance: Array = get_distance(
            x_positions_a=transmitter_x_position,
            y_positions_a=transmitter_y_position,
            x_positions_b=x_positions,
            y_positions_b=y_positions,
        )
        return jnp.where(
            distance < self.free_distance,
            20.0 * jnp.log10(wavelength / (4.0 * jnp.pi * distance)),
            20.0 * jnp.log10(wavelength / (4.0 * jnp.pi * self.free_distance))
            - 10.0
            * self.propagation_coefficient
            * jnp.log10(distance / self.free_distance),
        )

    @jax.jit
    def get_delta_tx(
        self: Self,
        transmitter_x_position: Array,
        transmitter_y_position: Array,
    ) -> Array:
        return get_distance(
            x_positions_a=self.init_transmitter_x_position,
            y_positions_a=self.init_transmitter_y_position,
            x_positions_b=transmitter_x_position,
            y_positions_b=transmitter_y_position,
        )

    @partial(jax.jit, static_argnums=(1,))
    def create_shadowing(
        self: Self,
        coordinate: Coordinate,
        transmitter_x_position: Array,
        transmitter_y_position: Array,
        key: Array,
    ) -> Array:
        delta_tx: Array = self.get_delta_tx(
            transmitter_x_position=transmitter_x_position,
            transmitter_y_position=transmitter_y_position,
        )
        delta_rx: Array = coordinate.get_delta_rx()
        correlation_matrix: Array = jnp.exp(
            -(delta_tx + delta_rx) / self.distance_correlation * jnp.log(2.0)
        )
        covariance_matrix: Array = correlation_matrix * self.standard_deviation**2.0
        l_covariance_matrix: Array = jnp.linalg.cholesky(covariance_matrix)
        normal_random: Array = random.normal(
            key=key, shape=(coordinate.x_mesh * coordinate.y_mesh,)
        )
        return (l_covariance_matrix @ normal_random).reshape(
            (coordinate.y_mesh, coordinate.x_mesh)
        )

    @partial(jax.jit, static_argnums=(5, 6, 7, 8))
    def get_channel_capacity(
        self: Self,
        receivers_key: Array,
        shadowing_key: Array,
        transmitter_x_index: Array,
        transmitter_y_index: Array,
        coordinate: Coordinate,
        receiver_number: int,
        noise_floor: float,
        bandwidth: float,
    ) -> Array:
        # 受信機は以下のようにして任意に設定することもできる
        # receivers = Receivers(
        #     x_positions=jnp.array([3.0,    7.0,    18.0  ]),
        #     y_positions=jnp.array([8.0,    13.0,   9.0   ]),
        #     noise_floor=jnp.array([-90.0,  -92.0,  -89.0 ]),
        #     bandwidth  =jnp.array([20.0e6, 40.0e6, 20.0e6]),
        # )
        receivers: Receivers = coordinate.create_random_position_receivers(
            key=receivers_key,
            number=receiver_number,
            noise_floor=noise_floor,
            bandwidth=bandwidth,
        )
        transmitter_x_position, transmitter_y_position = (
            coordinate.convert_indices_to_transmitter_positions(
                x_indices=transmitter_x_index,
                y_indices=transmitter_y_index,
            )
        )
        pathloss: Array = self.create_pathloss(
            coordinate=coordinate,
            transmitter_x_position=transmitter_x_position,
            transmitter_y_position=transmitter_y_position,
        )
        shadowing: Array = self.create_shadowing(
            coordinate=coordinate,
            transmitter_x_position=transmitter_x_position,
            transmitter_y_position=transmitter_y_position,
            key=shadowing_key,
        )
        receiver_x_indices, receiver_y_indices = (
            coordinate.convert_receiver_positions_to_indices(
                x_positions=receivers.x_positions,
                y_positions=receivers.y_positions,
            )
        )
        snr: Array = 10.0 ** (
            (
                (pathloss + shadowing).at[receiver_y_indices, receiver_x_indices].get()
                - receivers.noise_floor
            )
            / 10.0
        )
        return receivers.bandwidth * jnp.log2(1.0 + snr) * data_rate_unit

    @partial(jax.jit, static_argnums=(3, 4, 5, 6))
    def create_data_rate(
        self: Self,
        receivers_key: Array,
        shadowing_key: Array,
        coordinate: Coordinate,
        receiver_number: int,
        noise_floor: float,
        bandwidth: float,
    ) -> DataRate:
        data_rate: Array = jnp.zeros(
            (
                coordinate.y_mesh + 1,
                coordinate.x_mesh + 1,
                receiver_number,
            ),
            dtype=floating,
        )

        def fori_loop_vmap() -> Array:
            return jax.lax.fori_loop(
                lower=0,
                upper=coordinate.y_mesh + 1,
                body_fun=lambda y_index, data_rate: data_rate.at[y_index].set(
                    jax.vmap(
                        lambda x_index: self.get_channel_capacity(
                            transmitter_x_index=x_index,
                            transmitter_y_index=y_index,
                            coordinate=coordinate,
                            receiver_number=receiver_number,
                            noise_floor=noise_floor,
                            bandwidth=bandwidth,
                            receivers_key=receivers_key,
                            shadowing_key=shadowing_key,
                        )
                    )(jnp.arange(start=0, stop=coordinate.x_mesh + 1, step=1))
                ),
                init_val=data_rate,
            )

        def vmap_vmap() -> Array:
            return jax.vmap(
                lambda y_index: jax.vmap(
                    lambda x_index: self.get_channel_capacity(
                        transmitter_x_index=x_index,
                        transmitter_y_index=y_index,
                        coordinate=coordinate,
                        receiver_number=receiver_number,
                        noise_floor=noise_floor,
                        bandwidth=bandwidth,
                        receivers_key=receivers_key,
                        shadowing_key=shadowing_key,
                    )
                )(jnp.arange(start=0, stop=coordinate.x_mesh + 1, step=1))
            )(jnp.arange(start=0, stop=coordinate.y_mesh + 1, step=1))

        def fori_loop_fori_loop() -> Array:
            return jax.lax.fori_loop(
                lower=0,
                upper=coordinate.y_mesh + 1,
                body_fun=lambda y_index, data_rate_y: jax.lax.fori_loop(
                    lower=0,
                    upper=coordinate.x_mesh + 1,
                    body_fun=lambda x_index, data_rate_x: data_rate_x.at[
                        y_index, x_index
                    ].set(
                        self.get_channel_capacity(
                            transmitter_x_index=x_index,
                            transmitter_y_index=y_index,
                            coordinate=coordinate,
                            receiver_number=receiver_number,
                            noise_floor=noise_floor,
                            bandwidth=bandwidth,
                            receivers_key=receivers_key,
                            shadowing_key=shadowing_key,
                        )
                    ),
                    init_val=data_rate_y,
                ),
                init_val=data_rate,
            )

        return DataRate(
            data_rate=jax.lax.switch(
                index=data_rate_generation,
                branches=[
                    fori_loop_vmap,
                    vmap_vmap,
                    fori_loop_fori_loop,
                ],
            )
        )
