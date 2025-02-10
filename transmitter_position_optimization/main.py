import jax
import jax.numpy as jnp
from bayesian_optimization.acquisition import Acquisition

from jax._src.pjit import JitWrapped

from bayesian_optimization.kernel.exponential_kernel import (
    ExponentialTwoDimKernel,
)
from bayesian_optimization.kernel.gaussian_kernel import (
    GaussianTwoDimKernel,
)
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.kernel.matern3_kernel import (
    Matern3TwoDimKernel,
)
from bayesian_optimization.kernel.matern5_kernel import (
    Matern5TwoDimKernel,
)
from bayesian_optimization.parameter_optimization.log_random_search import (
    LogRandomSearch,
)
from bayesian_optimization.parameter_optimization.mcmc import (
    MCMC,
)

from constant import floating, integer
from environment.coordinate import Coordinate
from environment.evaluation import Evaluation
from environment.propagation import Propagation

from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from simulation.single import (
    single_transmitter_bo_simulation,
    single_transmitter_de_simulation,
    single_transmitter_rs_simulation,
)

if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    # jax.config.update("jax_platforms", "cpu")
    jax.config.update(
        "jax_enable_x64",
        integer == jnp.int64 and floating == jnp.float64,
    )

    coordinate = Coordinate(
        x_size=20.0,
        y_size=20.0,
        x_mesh=40,
        y_mesh=40,
    )
    propagation = Propagation(
        free_distance=1.0,
        propagation_coefficient=3.0,
        distance_correlation=10.0,
        standard_deviation=8.0,
        frequency=2.4e9,
        init_transmitter_x_position=jnp.asarray(coordinate.x_size / 2.0),
        init_transmitter_y_position=jnp.asarray(coordinate.y_size / 2.0),
    )
    receiver_number: int = 5
    noise_floor: float = -90.0
    bandwidth: float = 1.0e6
    simulation_number: int = 1000
    acquisition_function: JitWrapped = Acquisition.ucb()
    evaluation_function: JitWrapped = Evaluation.min
    init_train_indices_type: str = "random"

    single_transmitter_rs_simulation(
        propagation=propagation,
        coordinate=coordinate,
        transmitter_number=coordinate.x_mesh,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
        evaluation_function=evaluation_function,
        simulation_number=simulation_number,
    )
    single_transmitter_de_simulation(
        propagation=propagation,
        coordinate=coordinate,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
        evaluation_function=evaluation_function,
        simulation_number=simulation_number,
    )

    kernel: Kernel = GaussianTwoDimKernel()
    lower_bound, upper_bound = GaussianTwoDimKernel.log_random_search_range()
    parameter_optimization: ParameterOptimization = MCMC(
        std_params=lambda sp: sp / 100.0,
        count=1000,
        seed=0,
        parameter_optimization=MCMC(
            std_params=lambda sp: sp,
            count=1000,
            seed=0,
            parameter_optimization=LogRandomSearch(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                count=10000,
                seed=0,
            ),
        ),
    )

    single_transmitter_bo_simulation(
        propagation=propagation,
        coordinate=coordinate,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
        kernel=kernel,
        parameter_optimization=parameter_optimization,
        evaluation_function=evaluation_function,
        acquisition_function=acquisition_function,
        simulation_number=simulation_number,
        init_train_indices_type=init_train_indices_type,
        init_train_indices_random_number=4,
        init_train_indices_grid_number=2,
    )

    kernel: Kernel = Matern5TwoDimKernel()
    lower_bound, upper_bound = Matern5TwoDimKernel.log_random_search_range()
    parameter_optimization: ParameterOptimization = MCMC(
        std_params=lambda sp: sp / 100.0,
        count=1000,
        seed=0,
        parameter_optimization=MCMC(
            std_params=lambda sp: sp,
            count=1000,
            seed=0,
            parameter_optimization=LogRandomSearch(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                count=10000,
                seed=0,
            ),
        ),
    )

    single_transmitter_bo_simulation(
        propagation=propagation,
        coordinate=coordinate,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
        kernel=kernel,
        parameter_optimization=parameter_optimization,
        evaluation_function=evaluation_function,
        acquisition_function=acquisition_function,
        simulation_number=simulation_number,
        init_train_indices_type=init_train_indices_type,
        init_train_indices_random_number=4,
        init_train_indices_grid_number=2,
    )

    kernel: Kernel = Matern3TwoDimKernel()
    lower_bound, upper_bound = Matern3TwoDimKernel.log_random_search_range()
    parameter_optimization: ParameterOptimization = MCMC(
        std_params=lambda sp: sp / 100.0,
        count=1000,
        seed=0,
        parameter_optimization=MCMC(
            std_params=lambda sp: sp,
            count=1000,
            seed=0,
            parameter_optimization=LogRandomSearch(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                count=10000,
                seed=0,
            ),
        ),
    )

    single_transmitter_bo_simulation(
        propagation=propagation,
        coordinate=coordinate,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
        kernel=kernel,
        parameter_optimization=parameter_optimization,
        evaluation_function=evaluation_function,
        acquisition_function=acquisition_function,
        simulation_number=simulation_number,
        init_train_indices_type=init_train_indices_type,
        init_train_indices_random_number=4,
        init_train_indices_grid_number=2,
    )

    kernel: Kernel = ExponentialTwoDimKernel()
    lower_bound, upper_bound = ExponentialTwoDimKernel.log_random_search_range()
    parameter_optimization: ParameterOptimization = MCMC(
        std_params=lambda sp: sp / 100.0,
        count=1000,
        seed=0,
        parameter_optimization=MCMC(
            std_params=lambda sp: sp,
            count=1000,
            seed=0,
            parameter_optimization=LogRandomSearch(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                count=10000,
                seed=0,
            ),
        ),
    )

    single_transmitter_bo_simulation(
        propagation=propagation,
        coordinate=coordinate,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
        kernel=kernel,
        parameter_optimization=parameter_optimization,
        evaluation_function=evaluation_function,
        acquisition_function=acquisition_function,
        simulation_number=simulation_number,
        init_train_indices_type=init_train_indices_type,
        init_train_indices_random_number=4,
        init_train_indices_grid_number=2,
    )
