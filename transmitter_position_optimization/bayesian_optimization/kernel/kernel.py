from abc import ABCMeta, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
from jax import Array


@jax.tree_util.register_pytree_node_class
class Kernel(metaclass=ABCMeta):
    def tree_flatten(self: Self) -> tuple[tuple, dict]:
        return (
            (),
            {},
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "Kernel":
        return cls(*children, **aux_data)

    @staticmethod
    @abstractmethod
    @jax.jit
    def random_search_range() -> Array:
        pass

    @staticmethod
    @abstractmethod
    @jax.jit
    def log_random_search_range() -> Array:
        pass

    @abstractmethod
    @jax.jit
    def function(
        self: Self,
        input1: Array,
        input2: Array,
        parameter: Array,
    ) -> Array:
        pass

    @abstractmethod
    @jax.jit
    def gradient(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        pass

    @abstractmethod
    @jax.jit
    def hessian_matrix(
        self: Self,
        input1: Array,
        input2: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        pass

    @jax.jit
    def delta(self: Self, x: Array) -> Array:
        return jnp.where(x == 0.0, 1.0, 0.0)

    @jax.jit
    def create_k(
        self: Self,
        input_train_data: Array,
        parameter: Array,
    ) -> Array:
        return self.function(
            jax.vmap(lambda x: jnp.expand_dims(x, axis=0))(input_train_data),
            jax.vmap(lambda x: jnp.expand_dims(x, axis=1))(input_train_data),
            parameter,
        )

    @jax.jit
    def create_k_star(
        self: Self,
        input_train_data: Array,
        input_test_data: Array,
        parameter: Array,
    ) -> Array:
        return self.function(
            jax.vmap(lambda x: jnp.expand_dims(x, axis=0))(input_train_data),
            jax.vmap(lambda x: jnp.expand_dims(x, axis=1))(input_test_data),
            parameter,
        )

    @jax.jit
    def create_k_star_star(
        self: Self,
        input_test_data: Array,
        parameter: Array,
    ) -> Array:
        return self.function(
            jax.vmap(lambda x: jnp.expand_dims(x, axis=0))(input_test_data),
            jax.vmap(lambda x: jnp.expand_dims(x, axis=1))(input_test_data),
            parameter,
        )

    @jax.jit
    def create_gradient(
        self: Self,
        input_train_data: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return self.gradient(
            input1=jax.vmap(lambda x: jnp.expand_dims(x, axis=0))(input_train_data),
            input2=jax.vmap(lambda x: jnp.expand_dims(x, axis=1))(input_train_data),
            output_train_data=output_train_data,
            k_inv=k_inv,
            parameter=parameter,
        )

    @jax.jit
    def create_hessian_matrix(
        self: Self,
        input_train_data: Array,
        output_train_data: Array,
        k_inv: Array,
        parameter: Array,
    ) -> Array:
        return self.hessian_matrix(
            input1=jax.vmap(lambda x: jnp.expand_dims(x, axis=0))(input_train_data),
            input2=jax.vmap(lambda x: jnp.expand_dims(x, axis=1))(input_train_data),
            output_train_data=output_train_data,
            k_inv=k_inv,
            parameter=parameter,
        )

    @jax.jit
    def get_del_log_likelihood(
        self: Self,
        k_inv: Array,
        output_train_data: Array,
        del_k_del_parameter: Array,
    ) -> Array:
        return -jnp.trace(k_inv @ del_k_del_parameter) + (
            (k_inv @ output_train_data).T
            @ del_k_del_parameter
            @ (k_inv @ output_train_data)
        )

    @jax.jit
    def get_del_squared_log_likelihood(
        self: Self,
        k_inv: Array,
        output_train_data: Array,
        del_k_del_parameter: Array,
        del_k_del_parameter_star: Array,
        del_squared_k: Array,
    ) -> Array:
        return (
            jnp.trace(k_inv @ del_k_del_parameter_star @ k_inv @ del_k_del_parameter)
            - jnp.trace(k_inv @ del_squared_k)
            - 2.0
            * output_train_data.T
            @ k_inv
            @ del_k_del_parameter_star
            @ k_inv
            @ del_k_del_parameter
            @ k_inv
            @ output_train_data
            + output_train_data.T @ k_inv @ del_squared_k @ k_inv @ output_train_data
        )
