from typing import Literal

import jax.numpy as jnp
from bayesian_optimization.acquisition import Acquisition
from bayesian_optimization.kernel.exponential_kernel import (
    ExponentialPlusExponentialFourDimKernel,
    ExponentialPlusExponentialPlusExponentialSixDimKernel,
    ExponentialPolynomialTwoDimKernel,
    ExponentialTimesExponentialFourDimKernel,
    ExponentialTimesExponentialTimesExponentialSixDimKernel,
    ExponentialTwoDimKernel,
)
from bayesian_optimization.kernel.gaussian_kernel import (
    GaussianPlusGaussianFourDimKernel,
    GaussianPlusGaussianPlusGaussianSixDimKernel,
    GaussianPolynomialTwoDimKernel,
    GaussianTimesGaussianFourDimKernel,
    GaussianTimesGaussianTimesGaussianSixDimKernel,
    GaussianTwoDimKernel,
)
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.kernel.matern3_kernel import (
    Matern3PlusMatern3FourDimKernel,
    Matern3PlusMatern3PlusMatern3SixDimKernel,
    Matern3PolynomialTwoDimKernel,
    Matern3TimesMatern3FourDimKernel,
    Matern3TimesMatern3TimesMatern3SixDimKernel,
    Matern3TwoDimKernel,
)
from bayesian_optimization.kernel.matern5_kernel import (
    Matern5PlusMatern5FourDimKernel,
    Matern5PlusMatern5PlusMatern5SixDimKernel,
    Matern5PolynomialTwoDimKernel,
    Matern5TimesMatern5FourDimKernel,
    Matern5TimesMatern5TimesMatern5SixDimKernel,
    Matern5TwoDimKernel,
)
from bayesian_optimization.kernel.rational_quadratic_kernel import (
    RationalQuadraticPlusRationalQuadraticFourDimKernel,
    RationalQuadraticPlusRationalQuadraticPlusRationalQuadraticSixDimKernel,
    RationalQuadraticPolynomialTwoDimKernel,
    RationalQuadraticTimesRationalQuadraticFourDimKernel,
    RationalQuadraticTimesRationalQuadraticTimesRationalQuadraticSixDimKernel,
    RationalQuadraticTwoDimKernel,
)
from bayesian_optimization.parameter_optimization.log_random_search import (
    LogRandomSearch,
)
from bayesian_optimization.parameter_optimization.mcmc import (
    MCMC,
)
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from environment.coordinate import Coordinate
from environment.evaluation import Evaluation
from environment.propagation import Propagation
from jax._src.pjit import JitWrapped
from simulation.double import (
    double_transmitter_bo_simulation,
    double_transmitter_rs_simulation,
)
from simulation.single import (
    single_transmitter_bo_simulation,
    single_transmitter_de_simulation,
    single_transmitter_rs_simulation,
)
from simulation.triple import (
    triple_transmitter_bo_simulation,
    triple_transmitter_rs_simulation,
)


def single_transmitter_simulations() -> None:
    # 座標の設定
    coordinate = Coordinate(
        # 大きさ[m]
        x_size=20.0,
        y_size=20.0,
        # 受信機のメッシュ数
        x_mesh=40,
        y_mesh=40,
    )
    propagation = Propagation(
        # 自由空間伝搬距離[m]
        free_distance=1.0,
        # 伝搬係数
        propagation_coefficient=3.0,
        # 相関距離
        distance_correlation=10.0,
        # シャドウイングの標準偏差
        standard_deviation=8.0,
        # 周波数[Hz]
        frequency=2.4e9,
        # 送信機の初期位置
        init_transmitter_x_position=jnp.asarray(coordinate.x_size / 2.0),
        init_transmitter_y_position=jnp.asarray(coordinate.y_size / 2.0),
    )
    # 受信機の数
    receiver_number: int = 5
    # ノイズフロア[dBm]
    noise_floor: float = -90.0
    # 帯域幅[Hz]
    bandwidth: float = 20.0e6
    # シミュレーション回数
    simulation_number: int = 1000
    # 獲得関数
    acquisition_function: JitWrapped = Acquisition.ucb()
    # 評価関数（最適な位置の基準）
    evaluation_function: JitWrapped = Evaluation.min
    # 送信機の初期位置のパターン（グリッド状に配置orランダムに配置）
    init_train_indices_type: Literal["random", "grid"] = "grid"
    # 送信機の初期位置をランダムしたときの送信機の数
    init_train_indices_random_number = 4
    # 送信機の初期位置をグリッド状にしたときの1辺の送信機の数
    init_train_indices_grid_number = 2

    # ランダムサーチによるシミュレーション
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
    # 受信機からの位置推定によるシミュレーション
    single_transmitter_de_simulation(
        propagation=propagation,
        coordinate=coordinate,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
        evaluation_function=evaluation_function,
        simulation_number=simulation_number,
    )

    # ガウスカーネル
    kernel: Kernel = GaussianTwoDimKernel()
    # 対数ランダムサーチの範囲
    lower_bound, upper_bound = GaussianTwoDimKernel.log_random_search_range()
    # パラメータ最適化（対数ランダムサーチ->MCMC->MCMC）
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

    # ベイズ最適化によるシミュレーション
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = RationalQuadraticTwoDimKernel()
    lower_bound, upper_bound = RationalQuadraticTwoDimKernel.log_random_search_range()
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = GaussianPolynomialTwoDimKernel(1)
    lower_bound, upper_bound = GaussianPolynomialTwoDimKernel.log_random_search_range()
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = Matern3PolynomialTwoDimKernel(1)
    lower_bound, upper_bound = Matern3PolynomialTwoDimKernel.log_random_search_range()
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = Matern5PolynomialTwoDimKernel(1)
    lower_bound, upper_bound = Matern5PolynomialTwoDimKernel.log_random_search_range()
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = ExponentialPolynomialTwoDimKernel(1)
    lower_bound, upper_bound = (
        ExponentialPolynomialTwoDimKernel.log_random_search_range()
    )
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = RationalQuadraticPolynomialTwoDimKernel(1)
    lower_bound, upper_bound = (
        RationalQuadraticPolynomialTwoDimKernel.log_random_search_range()
    )
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )


def double_transmitter_simulations() -> None:
    coordinate = Coordinate(
        x_size=20.0,
        y_size=20.0,
        x_mesh=20,
        y_mesh=20,
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
    bandwidth: float = 20.0e6
    simulation_number: int = 1000
    acquisition_function: JitWrapped = Acquisition.ucb()
    evaluation_function: JitWrapped = Evaluation.min
    init_train_indices_type: Literal["random", "grid"] = "grid"
    init_train_indices_random_number = 4**2
    init_train_indices_grid_number = 2

    double_transmitter_rs_simulation(
        propagation=propagation,
        coordinate=coordinate,
        transmitter_number=coordinate.x_mesh,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
        evaluation_function=evaluation_function,
        simulation_number=simulation_number,
    )

    kernel: Kernel = GaussianPlusGaussianFourDimKernel()
    lower_bound, upper_bound = (
        GaussianPlusGaussianFourDimKernel.log_random_search_range()
    )
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

    double_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = Matern5PlusMatern5FourDimKernel()
    lower_bound, upper_bound = Matern5PlusMatern5FourDimKernel.log_random_search_range()
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

    double_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = Matern3PlusMatern3FourDimKernel()
    lower_bound, upper_bound = Matern3PlusMatern3FourDimKernel.log_random_search_range()
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

    double_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = ExponentialPlusExponentialFourDimKernel()
    lower_bound, upper_bound = (
        ExponentialPlusExponentialFourDimKernel.log_random_search_range()
    )
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

    double_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = RationalQuadraticPlusRationalQuadraticFourDimKernel()
    lower_bound, upper_bound = (
        RationalQuadraticPlusRationalQuadraticFourDimKernel.log_random_search_range()
    )
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

    double_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = GaussianTimesGaussianFourDimKernel()
    lower_bound, upper_bound = (
        GaussianTimesGaussianFourDimKernel.log_random_search_range()
    )
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

    double_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = Matern3TimesMatern3FourDimKernel()
    lower_bound, upper_bound = (
        Matern3TimesMatern3FourDimKernel.log_random_search_range()
    )
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

    double_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = Matern5TimesMatern5FourDimKernel()
    lower_bound, upper_bound = (
        Matern5TimesMatern5FourDimKernel.log_random_search_range()
    )
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

    double_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = ExponentialTimesExponentialFourDimKernel()
    lower_bound, upper_bound = (
        ExponentialTimesExponentialFourDimKernel.log_random_search_range()
    )
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

    double_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = RationalQuadraticTimesRationalQuadraticFourDimKernel()
    lower_bound, upper_bound = (
        RationalQuadraticTimesRationalQuadraticFourDimKernel.log_random_search_range()
    )
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

    double_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )


def triple_transmitter_simulations() -> None:
    coordinate = Coordinate(
        x_size=20.0,
        y_size=20.0,
        x_mesh=20,
        y_mesh=20,
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
    bandwidth: float = 20.0e6
    simulation_number: int = 1000
    acquisition_function: JitWrapped = Acquisition.ucb()
    evaluation_function: JitWrapped = Evaluation.min
    init_train_indices_type: Literal["random", "grid"] = "grid"
    init_train_indices_random_number = 4**3
    init_train_indices_grid_number = 2

    triple_transmitter_rs_simulation(
        propagation=propagation,
        coordinate=coordinate,
        transmitter_number=coordinate.x_mesh,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
        evaluation_function=evaluation_function,
        simulation_number=simulation_number,
    )

    kernel: Kernel = GaussianPlusGaussianPlusGaussianSixDimKernel()
    lower_bound, upper_bound = (
        GaussianPlusGaussianPlusGaussianSixDimKernel.log_random_search_range()
    )
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

    triple_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = Matern5PlusMatern5PlusMatern5SixDimKernel()
    lower_bound, upper_bound = (
        Matern5PlusMatern5PlusMatern5SixDimKernel.log_random_search_range()
    )
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

    triple_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = Matern3PlusMatern3PlusMatern3SixDimKernel()
    lower_bound, upper_bound = (
        Matern3PlusMatern3PlusMatern3SixDimKernel.log_random_search_range()
    )
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

    triple_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = ExponentialPlusExponentialPlusExponentialSixDimKernel()
    lower_bound, upper_bound = (
        ExponentialPlusExponentialPlusExponentialSixDimKernel.log_random_search_range()
    )
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

    triple_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = (
        RationalQuadraticPlusRationalQuadraticPlusRationalQuadraticSixDimKernel()
    )
    lower_bound, upper_bound = (
        RationalQuadraticPlusRationalQuadraticPlusRationalQuadraticSixDimKernel.log_random_search_range()
    )
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

    triple_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = GaussianTimesGaussianTimesGaussianSixDimKernel()
    lower_bound, upper_bound = (
        GaussianTimesGaussianTimesGaussianSixDimKernel.log_random_search_range()
    )
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

    triple_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = Matern3TimesMatern3TimesMatern3SixDimKernel()
    lower_bound, upper_bound = (
        Matern3TimesMatern3TimesMatern3SixDimKernel.log_random_search_range()
    )
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

    triple_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = Matern5TimesMatern5TimesMatern5SixDimKernel()
    lower_bound, upper_bound = (
        Matern5TimesMatern5TimesMatern5SixDimKernel.log_random_search_range()
    )
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

    triple_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = ExponentialTimesExponentialTimesExponentialSixDimKernel()
    lower_bound, upper_bound = (
        ExponentialTimesExponentialTimesExponentialSixDimKernel.log_random_search_range()
    )
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

    triple_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )

    kernel: Kernel = (
        RationalQuadraticTimesRationalQuadraticTimesRationalQuadraticSixDimKernel()
    )
    lower_bound, upper_bound = (
        RationalQuadraticTimesRationalQuadraticTimesRationalQuadraticSixDimKernel.log_random_search_range()
    )
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

    triple_transmitter_bo_simulation(
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
        init_train_indices_random_number=init_train_indices_random_number,
        init_train_indices_grid_number=init_train_indices_grid_number,
    )
