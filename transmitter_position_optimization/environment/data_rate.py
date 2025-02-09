from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from constant import (
    double_transmitter_max_size,
    double_transmitter_search_size,
    floating,
    integer,
    triple_transmitter_max_size,
    triple_transmitter_search_size,
)
from jax import Array
from jax._src.pjit import JitWrapped


@jax.tree_util.register_pytree_node_class
class DataRate:
    def __init__(self: Self, data_rate: Array) -> None:
        self.data_rate: Array = data_rate

    def tree_flatten(self: Self) -> tuple[tuple[Array], dict]:
        return (
            (self.data_rate,),
            {},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "DataRate":
        return cls(*children, **aux_data)

    def create_single_transmitter_function(self: Self) -> JitWrapped:
        @jax.jit
        def function(x_indices: Array, y_indices: Array) -> Array:
            return self.data_rate.at[y_indices, x_indices].get()

        return function

    def create_double_transmitter_function(self: Self) -> JitWrapped:
        @jax.jit
        def function(
            x_indices_a: Array,
            y_indices_a: Array,
            x_indices_b: Array,
            y_indices_b: Array,
        ) -> Array:
            data_rate_a: Array = self.data_rate.at[y_indices_a, x_indices_a].get()
            data_rate_b: Array = self.data_rate.at[y_indices_b, x_indices_b].get()
            return jnp.where(data_rate_a > data_rate_b, data_rate_a, data_rate_b)

        return function

    def create_triple_transmitter_function(self: Self) -> JitWrapped:
        @jax.jit
        def function(
            x_indices_a: Array,
            y_indices_a: Array,
            x_indices_b: Array,
            y_indices_b: Array,
            x_indices_c: Array,
            y_indices_c: Array,
        ) -> Array:
            data_rate_a: Array = self.data_rate.at[y_indices_a, x_indices_a].get()
            data_rate_b: Array = self.data_rate.at[y_indices_b, x_indices_b].get()
            data_rate_c: Array = self.data_rate.at[y_indices_c, x_indices_c].get()
            data_rate_ab: Array = jnp.where(
                data_rate_a > data_rate_b, data_rate_a, data_rate_b
            )
            return jnp.where(data_rate_ab > data_rate_c, data_rate_ab, data_rate_c)

        return function

    @partial(jax.jit, static_argnums=(1,))
    def get_true_single_transmitter_indices(
        self: Self,
        evaluation_function: JitWrapped,
    ) -> Array:
        value: Array = evaluation_function(data_rate=self.data_rate)
        true_value_indices: tuple[Array, ...] = jnp.unravel_index(
            indices=jnp.argmax(value),
            shape=value.shape,
        )
        return jnp.asarray(
            [true_value_indices[1], true_value_indices[0]], dtype=integer
        )

    @partial(jax.jit, static_argnums=(1,))
    def get_true_double_transmitter_indices(
        self: Self,
        evaluation_function: JitWrapped,
    ) -> Array:
        transmitter_x_mesh: int = self.data_rate.shape[1]
        transmitter_y_mesh: int = self.data_rate.shape[0]

        def search_true_position(i: Array) -> tuple[Array, Array, Array]:
            data_rate_i: Array = self.data_rate[
                i // transmitter_x_mesh, i % transmitter_x_mesh
            ]
            expanded_data_rate_i: Array = jnp.expand_dims(
                jnp.expand_dims(data_rate_i, axis=0), axis=0
            )
            true_data_rate: Array = jnp.where(
                expanded_data_rate_i > self.data_rate,
                expanded_data_rate_i,
                self.data_rate,
            )
            value: Array = evaluation_function(true_data_rate)
            true_value: Array = value.max()
            true_indices: Array = jnp.argwhere(
                value >= true_value,
                size=double_transmitter_search_size,
                fill_value=-1,
            )
            return true_value, true_indices.T[1], true_indices.T[0]

        each_true_value, each_true_x_indices, each_true_y_indices = jax.vmap(
            search_true_position
        )(jnp.arange(start=0, stop=transmitter_x_mesh * transmitter_y_mesh, step=1))

        true_value: Array = each_true_value.max()
        true_indices: Array = jnp.argwhere(
            each_true_value == true_value,
            size=double_transmitter_max_size,
            fill_value=-1,
        )

        def arrange_indices(
            true_value_index_j: Array,
            true_value_x_indices_i: Array,
            true_value_y_indices_i: Array,
        ) -> Array:
            true_value_x_indices_j: Array = jnp.full(
                true_value_x_indices_i.shape,
                true_value_index_j % transmitter_x_mesh,
            )
            true_value_y_indices_j: Array = jnp.full(
                true_value_x_indices_i.shape,
                true_value_index_j // transmitter_x_mesh,
            )

            return jnp.asarray(
                [
                    true_value_x_indices_i,
                    true_value_y_indices_i,
                    true_value_x_indices_j,
                    true_value_y_indices_j,
                ],
                dtype=integer,
            ).T

        return jax.vmap(arrange_indices)(
            true_indices.T[0],
            each_true_x_indices[true_indices.T[0]],
            each_true_y_indices[true_indices.T[0]],
        ).reshape(-1, 4)

    @partial(jax.jit, static_argnums=(1,))
    def get_true_triple_transmitter_indices(
        self: Self,
        evaluation_function: JitWrapped,
    ) -> Array:
        transmitter_x_mesh: int = self.data_rate.shape[1]
        transmitter_y_mesh: int = self.data_rate.shape[0]

        def search_true_position_ij(i: Array, j: Array) -> tuple[Array, Array, Array]:
            data_rate_i: Array = self.data_rate[
                i // transmitter_x_mesh, i % transmitter_x_mesh
            ]
            data_rate_j: Array = self.data_rate[
                j // transmitter_x_mesh, j % transmitter_x_mesh
            ]

            data_rate_ij: Array = jnp.where(
                data_rate_i > data_rate_j, data_rate_i, data_rate_j
            )
            max_data_rate: Array = jnp.where(
                jnp.expand_dims(jnp.expand_dims(data_rate_ij, axis=0), axis=0)
                > self.data_rate,
                jnp.expand_dims(jnp.expand_dims(data_rate_ij, axis=0), axis=0),
                self.data_rate,
            )

            value: Array = evaluation_function(max_data_rate)
            true_value: Array = value.max()
            true_indices: Array = jnp.argwhere(
                value >= true_value,
                size=triple_transmitter_search_size,
                fill_value=-1,
            )
            return true_value, true_indices.T[1], true_indices.T[0]

        def search_max_position_i(
            i: Array,
            value: tuple[Array, Array, Array],
        ) -> tuple[Array, Array, Array]:
            true_position_ij: tuple[Array, Array, Array] = jax.vmap(
                lambda j: search_true_position_ij(i, j)
            )(jnp.arange(start=0, stop=transmitter_x_mesh * transmitter_y_mesh, step=1))
            return (
                value[0].at[i].set(true_position_ij[0]),
                value[1].at[i].set(true_position_ij[1]),
                value[2].at[i].set(true_position_ij[2]),
            )

        each_true_value, each_true_x_indices, each_true_y_indices = jax.lax.fori_loop(
            lower=0,
            upper=transmitter_x_mesh * transmitter_y_mesh,
            body_fun=search_max_position_i,
            init_val=(
                jnp.zeros(
                    shape=(
                        transmitter_x_mesh * transmitter_y_mesh,
                        transmitter_x_mesh * transmitter_y_mesh,
                    ),
                    dtype=floating,
                ),
                jnp.zeros(
                    shape=(
                        transmitter_x_mesh * transmitter_y_mesh,
                        transmitter_x_mesh * transmitter_y_mesh,
                        triple_transmitter_search_size,
                    ),
                    dtype=integer,
                ),
                jnp.zeros(
                    shape=(
                        transmitter_x_mesh * transmitter_y_mesh,
                        transmitter_x_mesh * transmitter_y_mesh,
                        triple_transmitter_search_size,
                    ),
                    dtype=integer,
                ),
            ),
        )

        true_value: Array = each_true_value.max()
        true_value_indices: Array = jnp.argwhere(
            each_true_value == true_value,
            size=triple_transmitter_max_size,
            fill_value=-1,
        )

        def arrange_indices(
            true_value_index_i: Array,
            true_value_index_j: Array,
            true_value_x_indices_k: Array,
            true_value_y_indices_k: Array,
        ) -> Array:
            true_value_x_indices_i: Array = jnp.full(
                true_value_x_indices_k.shape,
                true_value_index_i % transmitter_x_mesh,
            )
            true_value_y_indices_i: Array = jnp.full(
                true_value_x_indices_k.shape,
                true_value_index_i // transmitter_x_mesh,
            )
            true_value_x_indices_j: Array = jnp.full(
                true_value_x_indices_k.shape,
                true_value_index_j % transmitter_x_mesh,
            )
            true_value_y_indices_j: Array = jnp.full(
                true_value_x_indices_k.shape,
                true_value_index_j // transmitter_x_mesh,
            )
            return jnp.asarray(
                [
                    true_value_x_indices_i,
                    true_value_y_indices_i,
                    true_value_x_indices_j,
                    true_value_y_indices_j,
                    true_value_x_indices_k,
                    true_value_y_indices_k,
                ],
                dtype=integer,
            ).T

        return jax.vmap(arrange_indices)(
            true_value_indices.T[0],
            true_value_indices.T[1],
            each_true_x_indices[true_value_indices.T[0], true_value_indices.T[1]],
            each_true_y_indices[true_value_indices.T[0], true_value_indices.T[1]],
        ).reshape(-1, 6)
