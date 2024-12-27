from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from constant import data_rate_generation, floating, integer
from environment.coordinate import Coordinate
from environment.data_rate import DataRate
from environment.receivers import Receivers
from environment.transmitter import Transmitter
from jax import Array, random


@jax.tree_util.register_pytree_node_class
class Propagation:
    def __init__(
        self,
        free_distance: float | Array,
        propagation_coefficient: float | Array,
        distance_correlation: float | Array,
        standard_deviation: float | Array,
        seed: int | Array,
    ) -> None:
        self.free_distance: Array = jnp.asarray(free_distance, dtype=floating)
        self.propagation_coefficient: Array = jnp.asarray(
            propagation_coefficient, dtype=floating
        )
        self.distance_correlation: Array = jnp.asarray(
            distance_correlation, dtype=floating
        )
        self.standard_deviation: Array = jnp.asarray(standard_deviation, dtype=floating)
        self.seed: Array = jnp.asarray(seed, dtype=integer)

    def tree_flatten(self) -> tuple[tuple[Array, Array, Array, Array, Array], None]:
        return (
            (
                self.free_distance,
                self.propagation_coefficient,
                self.distance_correlation,
                self.standard_deviation,
                self.seed,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "Propagation":
        return cls(*children)

    @partial(jax.jit, static_argnums=(0, 1))
    def update_seed(self: Self, seed: int) -> "Propagation":
        return Propagation(
            free_distance=self.free_distance,
            propagation_coefficient=self.propagation_coefficient,
            distance_correlation=self.distance_correlation,
            standard_deviation=self.standard_deviation,
            seed=seed,
        )

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def create_pathloss(
        self: Self,
        coordinate: Coordinate,
        transmitter: Transmitter,
    ) -> Array:
        x_positions, y_positions = coordinate.create_all_receiver_positions()
        distance: Array = transmitter.get_transmitter_distance(
            x_positions=x_positions,
            y_positions=y_positions,
        )
        return jnp.where(
            distance < self.free_distance,
            20.0 * jnp.log10(transmitter.wavelength / (4.0 * jnp.pi * distance)),
            20.0
            * jnp.log10(transmitter.wavelength / (4.0 * jnp.pi * self.free_distance))
            - 10.0
            * self.propagation_coefficient
            * jnp.log10(distance / self.free_distance),
        )

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def create_shadowing(
        self: Self,
        coordinate: Coordinate,
        transmitter: Transmitter,
    ) -> Array:
        key = random.key(self.seed)
        delta_tx: Array = transmitter.get_delta_tx()
        delta_rx: Array = coordinate.get_delta_rx()
        correlation_matrix: Array = jnp.exp(
            -(delta_tx + delta_rx) / self.distance_correlation * jnp.log(2.0)
        )
        covariance_matrix: Array = correlation_matrix * self.standard_deviation**2.0
        l_covariance_matrix: Array = jnp.linalg.cholesky(covariance_matrix)
        normal_random = random.normal(
            key=key, shape=(int(coordinate.x_mesh) * int(coordinate.y_mesh),)
        )
        return (l_covariance_matrix @ normal_random).reshape(
            (coordinate.y_mesh, coordinate.x_mesh)
        )

    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7))
    def get_channel_capacity(
        self: Self,
        x_index: Array,
        y_index: Array,
        coordinate: Coordinate,
        frequency: float,
        receivers: Receivers,
        init_x_position: float,
        init_y_position: float,
    ) -> Array:
        x_position, y_position = coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_index,
            y_indices=y_index,
        )
        transmitter = Transmitter(
            x_position=x_position,
            y_position=y_position,
            init_x_position=init_x_position,
            init_y_position=init_y_position,
            frequency=frequency,
        )
        pathloss: Array = self.create_pathloss(
            coordinate=coordinate,
            transmitter=transmitter,
        )
        shadowing: Array = self.create_shadowing(
            coordinate=coordinate,
            transmitter=transmitter,
        )
        x_index, y_index = coordinate.convert_receiver_positions_to_indices(
            x_positions=receivers.x_positions,
            y_positions=receivers.y_positions,
        )
        snr: Array = 10.0 ** (
            ((pathloss + shadowing).at[y_index, x_index].get() - receivers.noise_floor)
            / 10.0
        )
        return receivers.bandwidth * jnp.log2(1.0 + snr) / 1.0e6

    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
    def create_data_rate(
        self: Self,
        coordinate: Coordinate,
        receivers: Receivers,
        frequency: float,
        init_x_position: float,
        init_y_position: float,
    ) -> DataRate:
        data_rate: Array = jnp.zeros(
            (
                int(coordinate.y_mesh) + 1,
                int(coordinate.x_mesh) + 1,
                receivers.bandwidth.size,
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
                            x_index=x_index,
                            y_index=y_index,
                            coordinate=coordinate,
                            frequency=frequency,
                            receivers=receivers,
                            init_x_position=init_x_position,
                            init_y_position=init_y_position,
                        )
                    )(jnp.arange(start=0, stop=int(coordinate.x_mesh) + 1, step=1))
                ),
                init_val=data_rate,
            )

        def vmap_vmap() -> Array:
            return jax.vmap(
                lambda y_index: jax.vmap(
                    lambda x_index: self.get_channel_capacity(
                        x_index=x_index,
                        y_index=y_index,
                        coordinate=coordinate,
                        frequency=frequency,
                        receivers=receivers,
                        init_x_position=init_x_position,
                        init_y_position=init_y_position,
                    )
                )(jnp.arange(start=0, stop=int(coordinate.x_mesh) + 1, step=1))
            )(jnp.arange(start=0, stop=int(coordinate.y_mesh) + 1, step=1))

        def fori_loop_fori_loop() -> Array:
            return jax.lax.fori_loop(
                lower=0,
                upper=coordinate.y_mesh + 1,
                body_fun=lambda j, data_rate: jax.lax.fori_loop(
                    lower=0,
                    upper=coordinate.x_mesh + 1,
                    body_fun=lambda i, data_rate: data_rate.at[j, i].set(
                        self.get_channel_capacity(
                            x_index=i,
                            y_index=j,
                            coordinate=coordinate,
                            frequency=frequency,
                            receivers=receivers,
                            init_x_position=init_x_position,
                            init_y_position=init_y_position,
                        )
                    ),
                    init_val=data_rate,
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
