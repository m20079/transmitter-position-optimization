from functools import partial
from typing import Any, Self

import jax
import jax.numpy as jnp
from constant import floating, integer
from environment.distance import get_distance
from environment.receivers import Receivers
from jax import Array, random


@jax.tree_util.register_pytree_node_class
class Coordinate:
    def __init__(
        self: Self,
        x_size: float,
        y_size: float,
        x_mesh: int,
        y_mesh: int,
    ) -> None:
        self.x_size: float = x_size
        self.y_size: float = y_size
        self.x_mesh: int = x_mesh
        self.y_mesh: int = y_mesh

    def tree_flatten(self: Self) -> tuple[tuple[()], dict[str, Any]]:
        return (
            (),
            {
                "x_size": self.x_size,
                "y_size": self.y_size,
                "x_mesh": self.x_mesh,
                "y_mesh": self.y_mesh,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "Coordinate":
        return cls(*children, **aux_data)

    @jax.jit
    def convert_indices_to_receiver_positions(
        self: Self,
        x_indices: Array,
        y_indices: Array,
    ) -> tuple[Array, Array]:
        return (
            x_indices.astype(floating) * self.x_size / float(self.x_mesh)
            + (self.x_size / float(self.x_mesh) / 2.0),
            y_indices.astype(floating) * self.y_size / float(self.y_mesh)
            + (self.y_size / float(self.y_mesh) / 2.0),
        )

    @jax.jit
    def convert_indices_to_transmitter_positions(
        self: Self,
        x_indices: Array,
        y_indices: Array,
    ) -> tuple[Array, Array]:
        return (
            x_indices.astype(floating) * self.x_size / float(self.x_mesh),
            y_indices.astype(floating) * self.y_size / float(self.y_mesh),
        )

    @jax.jit
    def convert_transmitter_positions_to_indices(
        self: Self,
        x_positions: Array,
        y_positions: Array,
    ) -> Array:
        return jnp.asarray(
            [
                x_positions * float(self.x_mesh) / self.x_size,
                y_positions * float(self.y_mesh) / self.y_size,
            ]
        ).astype(integer)

    @jax.jit
    def convert_receiver_positions_to_indices(
        self: Self,
        x_positions: Array,
        y_positions: Array,
    ) -> Array:
        return jnp.asarray(
            [
                x_positions * float(self.x_mesh) / self.x_size - 0.5,
                y_positions * float(self.y_mesh) / self.y_size - 0.5,
            ]
        ).astype(integer)

    @partial(jax.jit, static_argnums=(2,))
    def create_random_receiver_positions(
        self: Self,
        key: Array,
        number: int,
    ) -> Array:
        x_key, y_key = random.split(key)
        return self.convert_indices_to_receiver_positions(
            x_indices=random.randint(
                key=x_key,
                shape=(number,),
                minval=0,
                maxval=self.x_mesh,
                dtype=integer,
            ),
            y_indices=random.randint(
                key=y_key,
                shape=(number,),
                minval=0,
                maxval=self.y_mesh,
                dtype=integer,
            ),
        )

    @partial(jax.jit, static_argnums=(2, 3, 4))
    def create_random_position_receivers(
        self: Self,
        key: Array,
        number: int,
        noise_floor: float,
        bandwidth: float,
    ) -> Receivers:
        x_positions, y_positions = self.create_random_receiver_positions(
            key=key,
            number=number,
        )
        return Receivers(
            x_positions=x_positions,
            y_positions=y_positions,
            noise_floor=jnp.full(number, noise_floor),
            bandwidth=jnp.full(number, bandwidth),
        )

    @jax.jit
    def create_all_receiver_positions(
        self: Self,
    ) -> Array:
        x_positions, y_positions = self.convert_indices_to_receiver_positions(
            x_indices=jnp.arange(0, stop=int(self.x_mesh), step=1, dtype=integer),
            y_indices=jnp.arange(0, stop=int(self.y_mesh), step=1, dtype=integer),
        )
        return jnp.asarray(
            jnp.meshgrid(
                x_positions,
                y_positions,
            ),
            dtype=floating,
        )

    @jax.jit
    def get_delta_rx(
        self: Self,
    ) -> Array:
        x_positions, y_positions = self.create_all_receiver_positions()
        return get_distance(
            x_positions_a=x_positions.ravel()[None, :],
            y_positions_a=y_positions.ravel()[None, :],
            x_positions_b=x_positions.ravel()[:, None],
            y_positions_b=y_positions.ravel()[:, None],
        )

    @jax.jit
    def create_all_transmitter_positions(
        self: Self,
    ) -> Array:
        x_positions, y_positions = self.convert_indices_to_transmitter_positions(
            x_indices=jnp.arange(0, stop=int(self.x_mesh) + 1, step=1, dtype=integer),
            y_indices=jnp.arange(0, stop=int(self.y_mesh) + 1, step=1, dtype=integer),
        )
        return jnp.asarray(
            jnp.meshgrid(
                x_positions,
                y_positions,
            ),
            dtype=floating,
        )

    @partial(jax.jit, static_argnums=(2,))
    def create_random_transmitter_indices(
        self: Self,
        key: Array,
        number: int,
    ) -> Array:
        x_key, y_key = random.split(key)
        return jnp.asarray(
            [
                random.randint(
                    key=x_key,
                    shape=(number,),
                    minval=0,
                    maxval=self.x_mesh + 1,
                    dtype=integer,
                ),
                random.randint(
                    key=y_key,
                    shape=(number,),
                    minval=0,
                    maxval=self.y_mesh + 1,
                    dtype=integer,
                ),
            ],
            dtype=integer,
        )

    @partial(jax.jit, static_argnums=(1,))
    def create_grid_transmitter_indices(
        self: Self,
        number: int,
    ) -> Array:
        x_grid_size: int = self.x_mesh // (number * 2)
        y_grid_size: int = self.y_mesh // (number * 2)
        x_grid: Array = jnp.arange(1, number * 2, 2) * x_grid_size
        y_grid: Array = jnp.arange(1, number * 2, 2) * y_grid_size
        return jnp.asarray(jnp.meshgrid(x_grid, y_grid), dtype=integer).reshape(2, -1)

    @partial(jax.jit, static_argnums=(2,))
    def create_random_transmitter_positions(
        self: Self,
        key: Array,
        number: int,
    ) -> Array:
        x_indices, y_indices = self.create_random_transmitter_indices(key, number)
        return self.convert_indices_to_transmitter_positions(
            x_indices=x_indices,
            y_indices=y_indices,
        )

    @jax.jit
    def get_search_number(
        self: Self,
    ) -> int:
        return (self.x_mesh + 1) * (self.y_mesh + 1)

    def get_receivers_extent(
        self: Self,
    ) -> tuple[float, float, float, float]:
        return (
            0.0,
            self.x_size,
            self.y_size,
            0.0,
        )

    def get_transmitter_extent(
        self: Self,
    ) -> tuple[float, float, float, float]:
        x_half_size: float = self.x_size / float(self.x_mesh) * 0.5
        y_half_size: float = self.y_size / float(self.y_mesh) * 0.5
        return (
            -x_half_size,
            self.x_size + x_half_size,
            self.y_size + y_half_size,
            -y_half_size,
        )
