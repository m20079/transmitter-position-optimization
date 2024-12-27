from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from constant import floating, integer
from environment.distance import get_distance
from environment.receivers import Receivers
from jax import Array, random


@jax.tree_util.register_pytree_node_class
class Coordinate:
    def __init__(
        self: "Coordinate",
        x_size: float,
        y_size: float,
        x_mesh: int,
        y_mesh: int,
    ) -> None:
        self.x_size: Array = jnp.asarray(x_size, dtype=floating)
        self.y_size: Array = jnp.asarray(y_size, dtype=floating)
        self.x_mesh: Array = jnp.asarray(x_mesh, dtype=integer)
        self.y_mesh: Array = jnp.asarray(y_mesh, dtype=integer)

    def tree_flatten(self) -> tuple[tuple[Array, Array, Array, Array], None]:
        return ((self.x_size, self.y_size, self.x_mesh, self.y_mesh), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "Coordinate":
        return cls(*children)

    @partial(jax.jit, static_argnums=(0,))
    def convert_indices_to_receiver_positions(
        self: Self,
        x_indices: Array,
        y_indices: Array,
    ) -> Array:
        return jnp.asarray(
            [
                x_indices.astype(floating) * self.x_size / self.x_mesh.astype(floating)
                + (self.x_size / self.x_mesh.astype(floating) / 2.0),
                y_indices.astype(floating) * self.y_size / self.y_mesh.astype(floating)
                + (self.y_size / self.y_mesh.astype(floating) / 2.0),
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def convert_indices_to_transmitter_positions(
        self: Self,
        x_indices: Array,
        y_indices: Array,
    ) -> Array:
        return jnp.asarray(
            [
                x_indices.astype(floating) * self.x_size / self.x_mesh.astype(floating),
                y_indices.astype(floating) * self.y_size / self.y_mesh.astype(floating),
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def convert_transmitter_positions_to_indices(
        self: Self,
        x_positions: Array,
        y_positions: Array,
    ) -> Array:
        return jnp.asarray(
            [
                x_positions * self.x_mesh.astype(floating) / self.x_size,
                y_positions * self.y_mesh.astype(floating) / self.y_size,
            ]
        ).astype(integer)

    @partial(jax.jit, static_argnums=(0,))
    def convert_receiver_positions_to_indices(
        self: Self,
        x_positions: Array,
        y_positions: Array,
    ) -> Array:
        return jnp.asarray(
            [
                x_positions * self.x_mesh.astype(floating) / self.x_size - 0.5,
                y_positions * self.y_mesh.astype(floating) / self.y_size - 0.5,
            ]
        ).astype(integer)

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def create_random_receiver_positions(
        self: Self,
        seed: int,
        number: int,
    ) -> Array:
        key: Array = random.key(seed)
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

    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
    def create_random_receivers(
        self: Self,
        seed: int,
        number: int,
        noise_floor: float,
        bandwidth: float,
    ) -> Receivers:
        receiver_positions: Array = self.create_random_receiver_positions(
            seed=seed,
            number=number,
        )
        return Receivers(
            x_positions=receiver_positions.at[0].get(),
            y_positions=receiver_positions.at[1].get(),
            noise_floor=jnp.full(number, noise_floor),
            bandwidth=jnp.full(number, bandwidth),
        )

    @partial(jax.jit, static_argnums=(0,))
    def create_all_receiver_positions(
        self: Self,
    ) -> Array:
        receiver_positions: Array = self.convert_indices_to_receiver_positions(
            x_indices=jnp.arange(0, stop=int(self.x_mesh), step=1, dtype=integer),
            y_indices=jnp.arange(0, stop=int(self.y_mesh), step=1, dtype=integer),
        )
        return jnp.asarray(
            jnp.meshgrid(
                receiver_positions.at[0].get(),
                receiver_positions.at[1].get(),
            )
        )

    @partial(jax.jit, static_argnums=(0,))
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

    @partial(jax.jit, static_argnums=(0,))
    def create_all_transmitter_positions(
        self: Self,
    ) -> Array:
        transmitter_positions: Array = self.convert_indices_to_transmitter_positions(
            x_indices=jnp.arange(0, stop=int(self.x_mesh) + 1, step=1, dtype=integer),
            y_indices=jnp.arange(0, stop=int(self.y_mesh) + 1, step=1, dtype=integer),
        )
        return jnp.asarray(
            jnp.meshgrid(
                transmitter_positions.at[0].get(),
                transmitter_positions.at[1].get(),
            )
        )

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def create_random_transmitter_indices(
        self: Self,
        seed: int,
        number: int,
    ) -> Array:
        key: Array = random.key(seed)
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
            ]
        )

    @partial(jax.jit, static_argnums=(0, 1))
    def create_grid_transmitter_indices(
        self: Self,
        number: int,
    ) -> Array:
        x_grid_size: Array = self.x_mesh // (number * 2)
        y_grid_size: Array = self.y_mesh // (number * 2)
        x_grid: Array = jnp.arange(1, number * 2, 2) * x_grid_size
        y_grid: Array = jnp.arange(1, number * 2, 2) * y_grid_size
        return jnp.asarray(jnp.meshgrid(x_grid, y_grid)).reshape(2, -1)

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def create_random_transmitter_positions(
        self: Self,
        seed: int,
        number: int,
    ) -> Array:
        x_indices, y_indices = self.create_random_transmitter_indices(seed, number)
        return self.convert_indices_to_transmitter_positions(
            x_indices=x_indices,
            y_indices=y_indices,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_receivers_extent(
        self: Self,
    ) -> tuple[float, float, float, float]:
        return (
            0.0,
            float(self.x_size),
            float(self.y_size),
            0.0,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_transmitter_extent(
        self: Self,
    ) -> tuple[float, float, float, float]:
        x_half_size: Array = self.x_size / self.x_mesh.astype(floating) * 0.5
        y_half_size: Array = self.y_size / self.y_mesh.astype(floating) * 0.5
        return (
            -float(x_half_size),
            float(self.x_size + x_half_size),
            float(self.y_size + y_half_size),
            -float(y_half_size),
        )
