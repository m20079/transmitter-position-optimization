import constant
import jax
import jax.numpy as jnp
from bayesian_optimization.acquisition import (
    Acquisition,
)
from bayesian_optimization.kernel.gaussian_kernel import (
    DoubleGaussianTwoDimKernel,
    GaussianTwoDimKernel,
    TripleGaussianTwoDimKernel,
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
from graph import plot_heatmap_histogram, plot_reverse_heatmap_histogram
from log import log_all_result
from simulation import (
    double_transmitter_simulation,
    single_transmitter_simulation,
    triple_transmitter_simulation,
)


def double_gaussian_grid_min_ucb():
    debug_name = "double_gaussian_grid_am_ucb"
    coordinate = Coordinate(
        x_size=20.0,
        y_size=20.0,
        y_mesh=2,
        x_mesh=2,
    )
    propagation = Propagation(
        seed=0,
        free_distance=1.0,
        propagation_coefficient=3.0,
        distance_correlation=10.0,
        standard_deviation=8.0,
    )
    count, distance_error, each_distance_error, data_rate_error = (
        double_transmitter_simulation(
            coordinate=coordinate,
            propagation=propagation,
            simulation_count=10,
            receiver_number=5,
            noise_floor=-90.0,
            bandwidth=20.0e6,
            frequency=2.4e9,
            kernel=TripleGaussianTwoDimKernel(),
            parameter_optimization=MCMC(
                count=10,
                seed=0,
                sigma_params=jnp.asarray(
                    [1.0, 1.0, 1.0, 1.0, 0.00001], dtype=constant.floating
                ),
                parameter_optimization=MCMC(
                    count=10,
                    seed=0,
                    sigma_params=jnp.asarray(
                        [100.0, 100.0, 100.0, 100.0, 0.001],
                        dtype=constant.floating,
                    ),
                    parameter_optimization=RandomSearch(
                        count=10,
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
            evaluation_function=Evaluation.min,
            acquisition_function=Acquisition.ucb(),
            init_indices_pattern="grid",
            init_indices_number=2,
            debug_name=debug_name,
        )
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


if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    jax.config.update("jax_platforms", "cpu")
    # jax.config.update("jax_enable_x64", True)

    double_gaussian_grid_min_ucb()
