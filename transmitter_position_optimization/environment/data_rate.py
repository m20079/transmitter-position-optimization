from functools import partial
from typing import Self

import constant
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
    def __init__(
        self,
        data_rate: Array,
    ) -> None:
        self.data_rate: Array = data_rate

    def tree_flatten(self: Self) -> tuple[tuple[Array], None]:
        return (
            (self.data_rate,),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "DataRate":
        return cls(*children)

    def create_single_transmitter_function(
        self: Self,
    ) -> JitWrapped:
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

    @partial(jax.jit, static_argnums=(0, 1))
    def get_single_max_transmitter_indices(
        self: Self,
        evaluation_function: JitWrapped,
    ) -> Array:
        evaluation: Array = evaluation_function(data_rate=self.data_rate, axis=2)
        max_indices: tuple[Array, ...] = jnp.unravel_index(
            indices=jnp.argmax(evaluation),
            shape=evaluation.shape,
        )
        return jnp.asarray([max_indices[1], max_indices[0]], dtype=constant.integer)

    @partial(jax.jit, static_argnums=(0, 1))
    def get_double_max_transmitter_indices(
        self: Self,
        evaluation_function: JitWrapped,
    ) -> Array:
        transmitter_x_mesh: int = self.data_rate.shape[1]
        transmitter_y_mesh: int = self.data_rate.shape[0]

        def search_max_position_i(i: Array) -> tuple[Array, Array, Array]:
            data_rate_i: Array = self.data_rate[
                i // transmitter_x_mesh, i % transmitter_x_mesh
            ]
            expanded_data_rate_i: Array = jnp.expand_dims(
                jnp.expand_dims(data_rate_i, axis=0), axis=0
            )
            max_data_rate: Array = jnp.where(
                expanded_data_rate_i > self.data_rate,
                expanded_data_rate_i,
                self.data_rate,
            )
            evaluation: Array = evaluation_function(max_data_rate, axis=2)
            max_evaluation: Array = evaluation.max()
            max_indices: Array = jnp.argwhere(
                evaluation >= max_evaluation,
                size=double_transmitter_search_size,
                fill_value=-1,
            )
            return max_evaluation, max_indices.T[1], max_indices.T[0]

        max_each_evaluation, max_each_x_indices, max_each_y_indices = jax.vmap(
            search_max_position_i
        )(jnp.arange(start=0, stop=transmitter_x_mesh * transmitter_y_mesh, step=1))

        max_evaluation: Array = max_each_evaluation.max()
        max_evaluation_indices: Array = jnp.argwhere(
            max_each_evaluation == max_evaluation,
            size=double_transmitter_max_size,
            fill_value=-1,
        )

        def arrange_indices(
            max_evaluation_index_j: Array,
            max_evaluation_x_indices_i: Array,
            max_evaluation_y_indices_i: Array,
        ) -> Array:
            max_evaluation_x_indices_j: Array = jnp.full(
                max_evaluation_x_indices_i.shape,
                max_evaluation_index_j % transmitter_x_mesh,
            )
            max_evaluation_y_indices_j: Array = jnp.full(
                max_evaluation_x_indices_i.shape,
                max_evaluation_index_j // transmitter_x_mesh,
            )

            return jnp.asarray(
                [
                    max_evaluation_x_indices_i,
                    max_evaluation_y_indices_i,
                    max_evaluation_x_indices_j,
                    max_evaluation_y_indices_j,
                ],
                dtype=constant.integer,
            ).T

        return jax.vmap(arrange_indices)(
            max_evaluation_indices.T[0],
            max_each_x_indices[max_evaluation_indices.T[0]],
            max_each_y_indices[max_evaluation_indices.T[0]],
        ).reshape(-1, 4)

    @partial(jax.jit, static_argnums=(0, 1))
    def get_triple_max_transmitter_indices(
        self: Self,
        evaluation_function: JitWrapped,
    ) -> Array:
        transmitter_x_mesh: int = self.data_rate.shape[1]
        transmitter_y_mesh: int = self.data_rate.shape[0]

        def search_max_position_ij(i: Array, j: Array) -> tuple[Array, Array, Array]:
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

            evaluation: Array = evaluation_function(max_data_rate, axis=2)
            max_evaluation: Array = evaluation.max()
            max_indices: Array = jnp.argwhere(
                evaluation >= max_evaluation,
                size=triple_transmitter_search_size,
                fill_value=-1,
            )
            return max_evaluation, max_indices.T[1], max_indices.T[0]

        def search_max_position_i(
            i: Array,
            value: tuple[Array, Array, Array],
        ) -> tuple[Array, Array, Array]:
            max_position_ij: tuple[Array, Array, Array] = jax.vmap(
                lambda j: search_max_position_ij(i, j)
            )(jnp.arange(start=0, stop=transmitter_x_mesh * transmitter_y_mesh, step=1))
            return (
                value[0].at[i].set(max_position_ij[0]),
                value[1].at[i].set(max_position_ij[1]),
                value[2].at[i].set(max_position_ij[2]),
            )

        max_each_evaluation, max_each_x_indices, max_each_y_indices = jax.lax.fori_loop(
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

        max_evaluation: Array = max_each_evaluation.max()
        max_evaluation_indices: Array = jnp.argwhere(
            max_each_evaluation == max_evaluation,
            size=triple_transmitter_max_size,
            fill_value=-1,
        )

        def arrange_indices(
            max_evaluation_index_i: Array,
            max_evaluation_index_j: Array,
            max_evaluation_x_indices_k: Array,
            max_evaluation_y_indices_k: Array,
        ) -> Array:
            max_evaluation_x_indices_i: Array = jnp.full(
                max_evaluation_x_indices_k.shape,
                max_evaluation_index_i % transmitter_x_mesh,
            )
            max_evaluation_y_indices_i: Array = jnp.full(
                max_evaluation_x_indices_k.shape,
                max_evaluation_index_i // transmitter_x_mesh,
            )
            max_evaluation_x_indices_j: Array = jnp.full(
                max_evaluation_x_indices_k.shape,
                max_evaluation_index_j % transmitter_x_mesh,
            )
            max_evaluation_y_indices_j: Array = jnp.full(
                max_evaluation_x_indices_k.shape,
                max_evaluation_index_j // transmitter_x_mesh,
            )
            return jnp.asarray(
                [
                    max_evaluation_x_indices_i,
                    max_evaluation_y_indices_i,
                    max_evaluation_x_indices_j,
                    max_evaluation_y_indices_j,
                    max_evaluation_x_indices_k,
                    max_evaluation_y_indices_k,
                ],
                dtype=constant.integer,
            ).T

        return jax.vmap(arrange_indices)(
            max_evaluation_indices.T[0],
            max_evaluation_indices.T[1],
            max_each_x_indices[
                max_evaluation_indices.T[0], max_evaluation_indices.T[1]
            ],
            max_each_y_indices[
                max_evaluation_indices.T[0], max_evaluation_indices.T[1]
            ],
        ).reshape(-1, 6)
