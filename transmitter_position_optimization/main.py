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
from bayesian_optimization.kernel.kernel import Kernel
from jax._src.pjit import JitWrapped

from bayesian_optimization.kernel.exponential_kernel import DoubleExponentialTwoDimKernel, ExponentialTwoDimKernel
from bayesian_optimization.kernel.matern3_kernel import DoubleMatern3TwoDimKernel, Matern3TwoDimKernel
from bayesian_optimization.kernel.matern5_kernel import DoubleMatern5TwoDimKernel, Matern5TwoDimKernel

def double_grid(
        kernel: Kernel,
        kernel_name: str,
        evaluation: JitWrapped,
        evaluation_name: str,
        acquisition: JitWrapped,
        acquisition_name: str,
):
    debug_name = f"single_grid_{kernel_name}_{evaluation_name}_{acquisition_name}"
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
    count, distance_error,  data_rate_error = (
        single_transmitter_simulation(
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
                    [1.0, 1.0,1.0, 1.0, 0.00001], dtype=constant.floating
                ),
                parameter_optimization=MCMC(
                    count=1000,
                    seed=0,
                    sigma_params=jnp.asarray(
                        [100.0, 100.0,100.0, 100.0, 0.001],
                        dtype=constant.floating,
                    ),
                    parameter_optimization=RandomSearch(
                        count=10000,
                        seed=0,
                        lower_bound=jnp.asarray(
                            [0.0, 0.0,0.0, 0.0, 0.0], dtype=constant.floating
                        ),
                        upper_bound=jnp.asarray(
                            [10000.0, 10000.0,10000.0, 10000.0, 0.1],
                            dtype=constant.floating,
                        ),
                    ),
                ),
            ),
            evaluation_function=evaluation,
            acquisition_function=acquisition,
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
    # jax.config.update("jax_platforms", "cpu")
    # jax.config.update("jax_enable_x64", True)

    double_grid(kernel=DoubleGaussianTwoDimKernel(), kernel_name="gaussian", evaluation=Evaluation.min, evaluation_name="min", acquisition=Acquisition.ucb(), acquisition_name="ucb")
    double_grid(kernel=DoubleMatern3TwoDimKernel(), kernel_name="matern3", evaluation=Evaluation.min, evaluation_name="min", acquisition=Acquisition.ucb(), acquisition_name="ucb")
    double_grid(kernel=DoubleMatern5TwoDimKernel(), kernel_name="matern5", evaluation=Evaluation.min, evaluation_name="min", acquisition=Acquisition.ucb(), acquisition_name="ucb")
    double_grid(kernel=DoubleExponentialTwoDimKernel(), kernel_name="exp", evaluation=Evaluation.min, evaluation_name="min", acquisition=Acquisition.ucb(), acquisition_name="ucb")
