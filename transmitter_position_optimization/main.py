from typing import Literal

import constant
import jax
import jax.numpy as jnp
from bayesian_optimization.acquisition import (
    Acquisition,
)
from bayesian_optimization.kernel.exponential_kernel import (
    DoubleExponentialTwoDimKernel,
    ExponentialTwoDimKernel,
)
from bayesian_optimization.kernel.gaussian_kernel import (
    DoubleGaussianTwoDimKernel,
    GaussianTwoDimKernel,
    TripleGaussianTwoDimKernel,
)
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.kernel.matern3_kernel import (
    DoubleMatern3TwoDimKernel,
    Matern3TwoDimKernel,
)
from bayesian_optimization.kernel.matern5_kernel import (
    DoubleMatern5TwoDimKernel,
    Matern5TwoDimKernel,
)
from bayesian_optimization.parameter_optimization.mcmc import (
    MCMC,
)
from bayesian_optimization.parameter_optimization.random_search import (
    RandomSearch,
)
from environment.coordinate import Coordinate
from environment.evaluation import Evaluation
from environment.propagation import Propagation
from graph import (
    plot_box,
    plot_heatmap_histogram,
    plot_reverse_heatmap_histogram,
    plot_scatter_density,
)
from jax._src.pjit import JitWrapped
from log import log_all_result
from simulation import (
    double_transmitter_random_simulation,
    double_transmitter_simulation,
    single_transmitter_random_simulation,
    single_transmitter_simulation,
    triple_transmitter_random_simulation,
    triple_transmitter_simulation,
)


def single_random_simulation(
    evaluation: JitWrapped,
    evaluation_name: str,
) -> None:
    debug_name: str = f"single_random_{evaluation_name}"
    coordinate = Coordinate(
        x_size=20.0,
        y_size=20.0,
        y_mesh=40,
        x_mesh=40,
    )
    propagation = Propagation(
        seed=0,
        free_distance=1.0,
        propagation_coefficient=3.0,
        distance_correlation=10.0,
        standard_deviation=8.0,
    )
    (
        count,
        distance_error,
        data_rate_error,
    ) = single_transmitter_random_simulation(
        coordinate=coordinate,
        propagation=propagation,
        simulation_count=1000,
        receiver_number=5,
        noise_floor=-90.0,
        bandwidth=20.0e6,
        frequency=2.4e9,
        search_number=30,
        evaluation_function=evaluation,
        debug_name=debug_name,
    )
    log_all_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_error=data_rate_error,
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_de.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=distance_error,
        color_label="Distance Error",
        y_label="Frequency",
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_dre.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=data_rate_error,
        color_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_de_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=distance_error,
        x_label="Distance Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_dre_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=data_rate_error,
        x_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_box(
        file=f"{debug_name}_box.pdf",
        data=[
            count,
            distance_error,
            data_rate_error,
        ],
        x_tick_labels=[
            "Count",
            "Distance Error",
            "Data Rate Error",
        ],
        y_label="Value",
    )


def single_simulation(
    kernel: Kernel,
    kernel_name: str,
    evaluation: JitWrapped,
    evaluation_name: str,
    acquisition: JitWrapped,
    acquisition_name: str,
    pattern: Literal["grid", "random"],
) -> None:
    debug_name: str = (
        f"single_{pattern}_{kernel_name}_{evaluation_name}_{acquisition_name}"
    )
    coordinate = Coordinate(
        x_size=20.0,
        y_size=20.0,
        y_mesh=40,
        x_mesh=40,
    )
    propagation = Propagation(
        seed=0,
        free_distance=1.0,
        propagation_coefficient=3.0,
        distance_correlation=10.0,
        standard_deviation=8.0,
    )
    (
        count,
        distance_error,
        data_rate_error,
    ) = single_transmitter_simulation(
        coordinate=coordinate,
        propagation=propagation,
        simulation_count=1000,
        receiver_number=5,
        noise_floor=-90.0,
        bandwidth=20.0e6,
        frequency=2.4e9,
        kernel=kernel,
        parameter_optimization=MCMC(
            count=1000,
            seed=0,
            sigma_params=jnp.asarray([1.0, 1.0, 0.00001], dtype=constant.floating),
            parameter_optimization=MCMC(
                count=1000,
                seed=0,
                sigma_params=jnp.asarray(
                    [100.0, 100.0, 0.001], dtype=constant.floating
                ),
                parameter_optimization=RandomSearch(
                    count=100000,
                    seed=0,
                    lower_bound=jnp.asarray([0.0, 0.0, 0.0], dtype=constant.floating),
                    upper_bound=jnp.asarray(
                        [10000.0, 10000.0, 0.1],
                        dtype=constant.floating,
                    ),
                ),
            ),
        ),
        evaluation_function=evaluation,
        acquisition_function=acquisition,
        init_indices_pattern=pattern,
        init_indices_number=2 if pattern == "grid" else 4,
        debug_name=debug_name,
    )
    log_all_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_error=data_rate_error,
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_de.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=distance_error,
        color_label="Distance Error",
        y_label="Frequency",
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_dre.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=data_rate_error,
        color_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_de_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=distance_error,
        x_label="Distance Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_dre_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=data_rate_error,
        x_label="Data Rate Error",
        y_label="Frequency",
    )


def double_random_simulation(evaluation: JitWrapped, evaluation_name: str):
    debug_name: str = f"double_random_{evaluation_name}"
    coordinate = Coordinate(
        x_size=20.0,
        y_size=20.0,
        y_mesh=20,
        x_mesh=20,
    )
    propagation = Propagation(
        seed=0,
        free_distance=1.0,
        propagation_coefficient=3.0,
        distance_correlation=10.0,
        standard_deviation=8.0,
    )
    (
        count,
        distance_error,
        each_distance_error_a,
        each_distance_error_b,
        data_rate_error,
    ) = double_transmitter_random_simulation(
        coordinate=coordinate,
        propagation=propagation,
        simulation_count=1000,
        receiver_number=5,
        noise_floor=-90.0,
        bandwidth=20.0e6,
        frequency=2.4e9,
        search_number=30,
        evaluation_function=evaluation,
        debug_name=debug_name,
    )
    log_all_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_error=data_rate_error,
        each_distance_error_a=each_distance_error_a,
        each_distance_error_b=each_distance_error_b,
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_de.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=distance_error,
        color_label="Distance Error",
        y_label="Frequency",
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_dre.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=data_rate_error,
        color_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_de_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=distance_error,
        x_label="Distance Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_dre_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=data_rate_error,
        x_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_scatter_density(
        file=f"{debug_name}_scatter.pdf",
        x_value=each_distance_error_a,
        y_value=each_distance_error_b,
        color_label="Density",
    )
    plot_box(
        file=f"{debug_name}_box.pdf",
        data=[
            count,
            distance_error,
            data_rate_error,
            each_distance_error_a,
            each_distance_error_b,
        ],
        x_tick_labels=[
            "Count",
            "Distance Error",
            "Data Rate Error",
            "Distance Error A",
            "Distance Error B",
        ],
        y_label="Value",
    )


def double_simulation(
    kernel: Kernel,
    kernel_name: str,
    evaluation: JitWrapped,
    evaluation_name: str,
    acquisition: JitWrapped,
    acquisition_name: str,
    pattern: Literal["grid", "random"],
) -> None:
    debug_name = f"double_{pattern}_{kernel_name}_{evaluation_name}_{acquisition_name}"
    coordinate = Coordinate(
        x_size=20.0,
        y_size=20.0,
        y_mesh=20,
        x_mesh=20,
    )
    propagation = Propagation(
        seed=0,
        free_distance=1.0,
        propagation_coefficient=3.0,
        distance_correlation=10.0,
        standard_deviation=8.0,
    )
    (
        count,
        distance_error,
        each_distance_error_a,
        each_distance_error_b,
        data_rate_error,
    ) = double_transmitter_simulation(
        coordinate=coordinate,
        propagation=propagation,
        simulation_count=1000,
        receiver_number=5,
        noise_floor=-90.0,
        bandwidth=20.0e6,
        frequency=2.4e9,
        kernel=kernel,
        parameter_optimization=MCMC(
            count=1000,
            seed=0,
            sigma_params=jnp.asarray(
                [1.0, 1.0, 1.0, 1.0, 0.00001], dtype=constant.floating
            ),
            parameter_optimization=MCMC(
                count=1000,
                seed=0,
                sigma_params=jnp.asarray(
                    [100.0, 100.0, 100.0, 100.0, 0.001],
                    dtype=constant.floating,
                ),
                parameter_optimization=RandomSearch(
                    count=100000,
                    seed=0,
                    lower_bound=jnp.asarray(
                        [0.0, 0.0, 0.0, 0.0, 0.0], dtype=constant.floating
                    ),
                    upper_bound=jnp.asarray(
                        [10000.0, 10000.0, 10000.0, 10000.0, 0.1],
                        dtype=constant.floating,
                    ),
                ),
            ),
        ),
        evaluation_function=evaluation,
        acquisition_function=acquisition,
        init_indices_pattern=pattern,
        init_indices_number=2 if pattern == "grid" else 4,
        debug_name=debug_name,
    )
    log_all_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_error=data_rate_error,
        each_distance_error_a=each_distance_error_a,
        each_distance_error_b=each_distance_error_b,
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_de.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=distance_error,
        color_label="Distance Error",
        y_label="Frequency",
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_dre.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=data_rate_error,
        color_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_de_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=distance_error,
        x_label="Distance Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_dre_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=data_rate_error,
        x_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_scatter_density(
        file=f"{debug_name}_scatter.pdf",
        x_value=each_distance_error_a,
        y_value=each_distance_error_b,
        color_label="Density",
    )
    plot_box(
        file=f"{debug_name}_box.pdf",
        data=[
            count,
            distance_error,
            data_rate_error,
            each_distance_error_a,
            each_distance_error_b,
        ],
        x_tick_labels=[
            "Count",
            "Distance Error",
            "Data Rate Error",
            "Distance Error A",
            "Distance Error B",
        ],
        y_label="Value",
    )


def triple_random_simulation(evaluation: JitWrapped, evaluation_name: str):
    debug_name: str = f"triple_random_{evaluation_name}"
    coordinate = Coordinate(
        x_size=20.0,
        y_size=20.0,
        y_mesh=20,
        x_mesh=20,
    )
    propagation = Propagation(
        seed=0,
        free_distance=1.0,
        propagation_coefficient=3.0,
        distance_correlation=10.0,
        standard_deviation=8.0,
    )
    (
        count,
        distance_error,
        each_distance_error_a,
        each_distance_error_b,
        each_distance_error_c,
        data_rate_error,
    ) = triple_transmitter_random_simulation(
        coordinate=coordinate,
        propagation=propagation,
        simulation_count=1000,
        receiver_number=5,
        noise_floor=-90.0,
        bandwidth=20.0e6,
        frequency=2.4e9,
        search_number=30,
        evaluation_function=evaluation,
        debug_name=debug_name,
    )
    log_all_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_error=data_rate_error,
        each_distance_error_a=each_distance_error_a,
        each_distance_error_b=each_distance_error_b,
        each_distance_error_c=each_distance_error_c,
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_de.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=distance_error,
        color_label="Distance Error",
        y_label="Frequency",
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_dre.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=data_rate_error,
        color_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_de_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=distance_error,
        x_label="Distance Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_dre_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=data_rate_error,
        x_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_scatter_density(
        file=f"{debug_name}_scatter_ab.pdf",
        x_value=each_distance_error_a,
        y_value=each_distance_error_b,
        color_label="Frequency",
    )
    plot_scatter_density(
        file=f"{debug_name}_scatter_bc.pdf",
        x_value=each_distance_error_b,
        y_value=each_distance_error_c,
        color_label="Frequency",
    )
    plot_scatter_density(
        file=f"{debug_name}_scatter_ca.pdf",
        x_value=each_distance_error_c,
        y_value=each_distance_error_a,
        color_label="Frequency",
    )
    plot_box(
        file=f"{debug_name}_box.pdf",
        data=[
            count,
            distance_error,
            data_rate_error,
            each_distance_error_a,
            each_distance_error_b,
            each_distance_error_c,
        ],
        x_tick_labels=[
            "Count",
            "Distance Error",
            "Data Rate Error",
            "Distance Error A",
            "Distance Error B",
            "Distance Error C",
        ],
        y_label="Value",
    )


def triple_simulation(
    kernel: Kernel,
    kernel_name: str,
    evaluation: JitWrapped,
    evaluation_name: str,
    acquisition: JitWrapped,
    acquisition_name: str,
    pattern: Literal["grid", "random"],
) -> None:
    debug_name = f"triple_{pattern}_{kernel_name}_{evaluation_name}_{acquisition_name}"
    coordinate = Coordinate(
        x_size=20.0,
        y_size=20.0,
        y_mesh=20,
        x_mesh=20,
    )
    propagation = Propagation(
        seed=0,
        free_distance=1.0,
        propagation_coefficient=3.0,
        distance_correlation=10.0,
        standard_deviation=8.0,
    )
    (
        count,
        distance_error,
        each_distance_error_a,
        each_distance_error_b,
        each_distance_error_c,
        data_rate_error,
    ) = triple_transmitter_simulation(
        coordinate=coordinate,
        propagation=propagation,
        simulation_count=1000,
        receiver_number=5,
        noise_floor=-90.0,
        bandwidth=20.0e6,
        frequency=2.4e9,
        kernel=kernel,
        parameter_optimization=MCMC(
            count=1000,
            seed=0,
            sigma_params=jnp.asarray(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.00001], dtype=constant.floating
            ),
            parameter_optimization=MCMC(
                count=1000,
                seed=0,
                sigma_params=jnp.asarray(
                    [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.001],
                    dtype=constant.floating,
                ),
                parameter_optimization=RandomSearch(
                    count=100000,
                    seed=0,
                    lower_bound=jnp.asarray(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=constant.floating
                    ),
                    upper_bound=jnp.asarray(
                        [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 0.1],
                        dtype=constant.floating,
                    ),
                ),
            ),
        ),
        evaluation_function=evaluation,
        acquisition_function=acquisition,
        init_indices_pattern=pattern,
        init_indices_number=2 if pattern == "grid" else 4,
        debug_name=debug_name,
    )
    log_all_result(
        debug_name=debug_name,
        count=count,
        distance_error=distance_error,
        data_rate_error=data_rate_error,
        each_distance_error_a=each_distance_error_a,
        each_distance_error_b=each_distance_error_b,
        each_distance_error_c=each_distance_error_c,
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_de.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=distance_error,
        color_label="Distance Error",
        y_label="Frequency",
    )
    plot_heatmap_histogram(
        file=f"{debug_name}_dre.pdf",
        horizontal_value=count,
        x_label="Number of Measurements",
        color_value=data_rate_error,
        color_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_de_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=distance_error,
        x_label="Distance Error",
        y_label="Frequency",
    )
    plot_reverse_heatmap_histogram(
        file=f"{debug_name}_dre_r.pdf",
        color_value=count,
        color_label="Number of Measurements",
        horizontal_value=data_rate_error,
        x_label="Data Rate Error",
        y_label="Frequency",
    )
    plot_scatter_density(
        file=f"{debug_name}_scatter_ab.pdf",
        x_value=each_distance_error_a,
        y_value=each_distance_error_b,
        color_label="Frequency",
    )
    plot_scatter_density(
        file=f"{debug_name}_scatter_bc.pdf",
        x_value=each_distance_error_b,
        y_value=each_distance_error_c,
        color_label="Frequency",
    )
    plot_scatter_density(
        file=f"{debug_name}_scatter_ca.pdf",
        x_value=each_distance_error_c,
        y_value=each_distance_error_a,
        color_label="Frequency",
    )
    plot_box(
        file=f"{debug_name}_box.pdf",
        data=[
            count,
            distance_error,
            data_rate_error,
            each_distance_error_a,
            each_distance_error_b,
            each_distance_error_c,
        ],
        x_tick_labels=[
            "Count",
            "Distance Error",
            "Data Rate Error",
            "Distance Error A",
            "Distance Error B",
            "Distance Error C",
        ],
        y_label="Value",
    )


if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    # jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    double_random_simulation(
        evaluation=Evaluation.min,
        evaluation_name="min",
    )

    # double_simulation(
    #     kernel=DoubleGaussianTwoDimKernel(),
    #     kernel_name="gaussian",
    #     evaluation=Evaluation.min,
    #     evaluation_name="min",
    #     acquisition=Acquisition.ucb(),
    #     acquisition_name="ucb",
    #     pattern="grid",
    # )
    # double_simulation(
    #     kernel=DoubleMatern3TwoDimKernel(),
    #     kernel_name="matern3",
    #     evaluation=Evaluation.min,
    #     evaluation_name="min",
    #     acquisition=Acquisition.ucb(),
    #     acquisition_name="ucb",
    #     pattern="grid",
    # )
    # double_simulation(
    #     kernel=DoubleMatern5TwoDimKernel(),
    #     kernel_name="matern5",
    #     evaluation=Evaluation.min,
    #     evaluation_name="min",
    #     acquisition=Acquisition.ucb(),
    #     acquisition_name="ucb",
    #     pattern="grid",
    # )
    # double_simulation(
    #     kernel=DoubleExponentialTwoDimKernel(),
    #     kernel_name="exp",
    #     evaluation=Evaluation.min,
    #     evaluation_name="min",
    #     acquisition=Acquisition.ucb(),
    #     acquisition_name="ucb",
    #     pattern="grid",
    # )
