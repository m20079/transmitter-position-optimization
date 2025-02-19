from typing import Self

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from bayesian_optimization.gaussian_process_regression import (
    GaussianProcessRegression,
)
from bayesian_optimization.kernel.exponential_kernel import (
    ExponentialKernel,
)
from bayesian_optimization.kernel.gaussian_kernel import (
    GaussianKernel,
)
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.kernel.matern3_kernel import (
    Matern3Kernel,
)
from bayesian_optimization.kernel.matern5_kernel import (
    Matern5Kernel,
    Matern5PolynomialTwoDimKernel,
)
from bayesian_optimization.kernel.rational_quadratic_kernel import (
    RationalQuadraticKernel,
)
from bayesian_optimization.parameter_optimization.log_random_search import (
    LogRandomSearch,
)
from constant import floating, integer, platforms


@jax.tree_util.register_pytree_node_class
class PolynomialKernel(Kernel):
    def __init__(self: Self, power: int) -> None:
        self.power: int = power

    def tree_flatten(self: Self) -> tuple[tuple[()], dict[str, int]]:
        return (
            (),
            {
                "power": self.power,
            },
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> "PolynomialKernel":
        return cls(*children, **aux_data)

    @staticmethod
    @jax.jit
    def random_search_range() -> jax.Array:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0e4, 1.0e4, 1.0e4, 1.0e-1],
            ],
            dtype=floating,
        )

    @staticmethod
    @jax.jit
    def log_random_search_range() -> jax.Array:
        return jnp.asarray(
            [
                [1.0e-2],
                [1.0e4],
            ],
            dtype=floating,
        )

    @jax.jit
    def function(
        self: Self,
        input1: jax.Array,
        input2: jax.Array,
        parameter: jax.Array,
    ) -> jax.Array:
        input_abs: jax.Array = jnp.abs(input1[0] - input2[0])
        return (
            parameter[0]
            # * jnp.exp(-jnp.power(input1[0] - input2[0], 2) / parameter[1])
            # * parameter[2]
            # * jnp.exp(-jnp.power(input1[0] - input2[0], 2) / parameter[3])
            # + self.delta(input1[0] - input2[0]) * parameter[4]
        )

    @jax.jit
    def gradient(
        self: Self,
        input1: jax.Array,
        input2: jax.Array,
        output_train_data: jax.Array,
        k_inv: jax.Array,
        parameter: jax.Array,
    ) -> jax.Array:
        return jnp.asarray([])

    @jax.jit
    def hessian_matrix(
        self: Self,
        input1: jax.Array,
        input2: jax.Array,
        output_train_data: jax.Array,
        k_inv: jax.Array,
        parameter: jax.Array,
    ) -> jax.Array:
        return jnp.asarray([])


if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    jax.config.update("jax_platforms", platforms)
    jax.config.update(
        "jax_enable_x64",
        integer == jnp.int64 and floating == jnp.float64,
    )

    def original_function(x: jax.Array) -> jax.Array:
        return (
            1.9 * jnp.cos(1.9 * x + 3.0)
            + 1.3 * jnp.sin(1.2 * x + 4.0)
            - (x - 5) ** 2.0 / 20.0
            + 2.0
        )

    input = jnp.array([0.5, 2.0, 2.5, 4.0, 5.0, 8.0, 9.5])
    output = original_function(input)
    test = jnp.arange(0.0, 10.0, 0.01)

    kernel = PolynomialKernel(power=2)
    lower, upper = kernel.log_random_search_range()

    parameter_optimization = LogRandomSearch(
        lower_bound=lower,
        upper_bound=upper,
        count=10000,
        seed=19,
    )
    parameter = parameter_optimization.optimize(
        input_train_data=jnp.array([input]),
        output_train_data=output,
        kernel=kernel,
    )
    parameter.at[2].set(1.0)
    print(parameter)
    ppp = jnp.array(
        [parameter.at[0].get() ** 0.0, parameter.at[1].get(), parameter.at[1].get()]
    )

    gaussian_process = GaussianProcessRegression(
        input_train_data=jnp.array([input]),
        output_train_data=output,
        kernel=kernel,
        parameter=ppp,
    )
    mean, std = gaussian_process.function(jnp.array([test]))

    plt.figure(figsize=(6.0 * 0.8, 4.0 * 0.8))
    plt.scatter(
        input,
        output,
        color="red",
        zorder=2,
    )
    plt.plot(
        test,
        mean,
        color="blue",
        zorder=1,
    )
    plt.fill_between(
        test,
        mean - std,
        mean + std,
        alpha=0.2,
        color="blue",
        zorder=1,
    )
    plt.tick_params(direction="in")
    plt.savefig("bias.svg", bbox_inches="tight")
    plt.clf()
    plt.close()
