import constant
import japanize_matplotlib
import jax
import jax.numpy as jnp
import matplotlib.font_manager
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from bayesian_optimization.acquisition import Acquisition
from bayesian_optimization.bayesian_optimization import (
    single_transmitter_bayesian_optimization,
)
from bayesian_optimization.gaussian_process_regression import (
    GaussianProcessRegression,
)
from bayesian_optimization.kernel.exponential_kernel import (
    ExponentialTwoDimKernel,
)
from bayesian_optimization.kernel.gaussian_kernel import (
    GaussianKernel,
    GaussianTwoDimKernel,
)
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.kernel.matern3_kernel import (
    Matern3TwoDimKernel,
)
from bayesian_optimization.kernel.matern5_kernel import (
    Matern5TwoDimKernel,
)
from bayesian_optimization.parameter_optimization.bfgs import (
    BFGS,
)
from bayesian_optimization.parameter_optimization.conjugate_gradient import (
    ConjugateGradient,
)
from bayesian_optimization.parameter_optimization.dfp import DFP
from bayesian_optimization.parameter_optimization.gradient_descent import (
    GradientDescent,
)
from bayesian_optimization.parameter_optimization.grid_search import (
    GridSearch,
)
from bayesian_optimization.parameter_optimization.log_random_search import (
    LogRandomSearch,
)
from bayesian_optimization.parameter_optimization.mcmc import (
    MCMC,
)
from bayesian_optimization.parameter_optimization.newton import (
    Newton,
)
from bayesian_optimization.parameter_optimization.random_search import (
    RandomSearch,
)
from constant import floating, integer
from environment.coordinate import Coordinate
from environment.evaluation import Evaluation
from environment.propagation import Propagation
from jax import random

if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    jax.config.update("jax_platforms", "cpu")
    jax.config.update(
        "jax_enable_x64",
        integer == jnp.int64 and floating == jnp.float64,
    )

    # import matplotlib
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from matplotlib.font_manager import FontProperties

    # fp = FontProperties(
    #     family="IPAexGothic",
    #     math_fontfamily="cm",
    #     size=16.0,
    # )

    # # データ準備
    # x = np.linspace(0, 10, 5)  # 横軸の描画範囲指定
    # y1 = 2 * x + 3  # 式1 y = 2x + 3より、縦軸の値算出
    # y2 = 3 * x + 1  # 式2 y = 3x + 1より、縦軸の値算出

    # # グラフの装飾
    # plt.title(
    #     r"日本語表示テスト abc $\mu y = ax + b \frac{\sigma}{x^{'}}$", fontproperties=fp
    # )  # タイトル
    # plt.xlabel("x軸", fontproperties=fp)  # x軸ラベル
    # plt.ylabel(r"y軸 $\mu$", fontproperties=fp)  # y軸ラベル

    # # グラフの描画
    # plt.plot(x, y1, label="式 y = 2x + 3")  # 式1の描画
    # plt.plot(x, y2, label="式 y = 3x + 1")  # 式2の描画
    # plt.legend(loc="upper left", prop=fp)  # 凡例表示
    # plt.show()

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
    # @jax.jit
    # def a(
    #     input_train_data,
    #     output_train_data,
    #     parameter,
    #     gradient,
    #     update_vector,
    #     log_likelihood,
    #     kernel,
    # ):
    #     return jnp.array(1.0)

    # parameter_optimization = BFGS(
    #     count=20,
    #     learning_rate=GradientDescent.backtracking(
    #         condition_type="wolfe", count=100000
    #     ),
    #     parameter_optimization=LogRandomSearch(
    #         lower_bound=jnp.asarray([0.1, 0.1, 0.00001], dtype=constant.floating),
    #         upper_bound=jnp.asarray([100000.0, 100000.0, 0.1], dtype=constant.floating),
    #         count=1000,
    #         seed=0,
    #     ),
    # )

    # Newton(
    #     count=1000,
    #     # learning_rate=jnp.asarray([0.1, 0.1, 0.0001], dtype=constant.floating),
    #     parameter_optimization=LogRandomSearch(
    #         lower_bound=jnp.asarray([0.1, 0.1, 0.00001], dtype=constant.floating),
    #         upper_bound=jnp.asarray([100000.0, 100000.0, 0.1], dtype=constant.floating),
    #         count=10,
    #         seed=0,
    #     ),
    # )
    # parameter: jax.Array = parameter_optimization.optimize(
    #     input_train_data=jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 7.0]]),
    #     output_train_data=jnp.array([1.0, 2.0, 3.0, 4.0, 2.0, 3.0]),
    #     kernel=GaussianKernel(),
    # )
    # print(parameter)
    # parameter: jax.Array = parameter_optimization.optimize(
    #     input_train_data=jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 7.0]]),
    #     output_train_data=jnp.array([1.0, 2.0, 3.0, 4.0, 2.0, 1.0]),
    #     kernel=GaussianKernel(),
    # )
    # print(parameter)
    # print(parameter)
    # gaussian_process = GaussianProcessRegression(
    #     input_train_data=jnp.array([[1.0, 2.0, 3.0]]),
    #     output_train_data=jnp.array([1.0, 2.0, 3.0]),
    #     kernel=GaussianKernel(),
    #     parameter=parameter,
    # )

    # pathloss = propagation.create_pathloss(
    #     coordinate=coordinate,
    #     transmitter_x_position=jnp.asarray(0.0),
    #     transmitter_y_position=jnp.asarray(0.0),
    # )
    # shadowing = propagation.create_shadowing(
    #     coordinate=coordinate,
    #     transmitter_x_position=jnp.asarray(0.0),
    #     transmitter_y_position=jnp.asarray(0.0),
    #     key=random.key(1),
    # )
    print(
        single_transmitter_bayesian_optimization(
            propagation=propagation,
            coordinate=coordinate,
            receiver_number=5,
            bandwidth=20.0e6,
            noise_floor=-90.0,
            receivers_key=random.key(1),
            shadowing_key=random.key(1),
            kernel=GaussianTwoDimKernel(),
            parameter_optimization=MCMC(
                count=10,
                seed=0,
                std_params=lambda _: jnp.asarray(
                    [1.0, 1.0, 0.00001], dtype=constant.floating
                ),
                parameter_optimization=MCMC(
                    count=1000,
                    seed=0,
                    std_params=lambda _: jnp.asarray(
                        [100.0, 100.0, 0.001], dtype=constant.floating
                    ),
                    parameter_optimization=RandomSearch(
                        count=10000,
                        seed=0,
                        lower_bound=jnp.asarray(
                            [0.0, 0.0, 0.0], dtype=constant.floating
                        ),
                        upper_bound=jnp.asarray(
                            [10000.0, 10000.0, 0.1],
                            dtype=constant.floating,
                        ),
                    ),
                ),
            ),
            evaluation_function=Evaluation.min,
            acquisition_function=Acquisition.ucb(),
            x_train_indices=jnp.asarray([5, 10, 5, 10]),
            y_train_indices=jnp.asarray([5, 10, 10, 5]),
        )
    )


# def single_random_simulation(
#     evaluation: JitWrapped,
#     evaluation_name: str,
#     search_number: int,
# ) -> None:
#     debug_name: str = f"single_random{search_number}_{evaluation_name}"
#     coordinate = Coordinate(
#         x_size=20.0,
#         y_size=20.0,
#         y_mesh=40,
#         x_mesh=40,
#     )
#     propagation = Propagation(
#         seed=0,
#         free_distance=1.0,
#         propagation_coefficient=3.0,
#         distance_correlation=10.0,
#         standard_deviation=8.0,
#     )
#     (
#         count,
#         distance_error,
#         data_rate_error,
#     ) = single_transmitter_random_simulation(
#         coordinate=coordinate,
#         propagation=propagation,
#         simulation_count=1000,
#         receiver_number=5,
#         noise_floor=-90.0,
#         bandwidth=20.0e6,
#         frequency=2.4e9,
#         search_number=search_number,
#         evaluation_function=evaluation,
#         debug_name=debug_name,
#     )
#     log_all_result(
#         debug_name=debug_name,
#         count=count,
#         distance_error=distance_error,
#         data_rate_error=data_rate_error,
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_de.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=distance_error,
#         color_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_dre.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=data_rate_error,
#         color_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_de_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=distance_error,
#         x_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_dre_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=data_rate_error,
#         x_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_box(
#         file=f"{debug_name}_box.pdf",
#         data=[
#             count,
#             distance_error,
#             data_rate_error,
#         ],
#         x_tick_labels=[
#             "Count",
#             "Distance Error",
#             "Data Rate Error",
#         ],
#         y_label="Value",
#     )


# def single_simulation(
#     kernel: Kernel,
#     kernel_name: str,
#     evaluation: JitWrapped,
#     evaluation_name: str,
#     acquisition: JitWrapped,
#     acquisition_name: str,
#     pattern: Literal["grid", "random"],
# ) -> None:
#     debug_name: str = (
#         f"single_{pattern}_{kernel_name}_{evaluation_name}_{acquisition_name}"
#     )
#     coordinate = Coordinate(
#         x_size=20.0,
#         y_size=20.0,
#         y_mesh=10,
#         x_mesh=10,
#     )
#     propagation = Propagation(
#         seed=0,
#         free_distance=1.0,
#         propagation_coefficient=3.0,
#         distance_correlation=10.0,
#         standard_deviation=8.0,
#     )
#     (
#         count,
#         distance_error,
#         data_rate_error,
#     ) = single_transmitter_simulation(
#         coordinate=coordinate,
#         propagation=propagation,
#         simulation_count=10,
#         receiver_number=5,
#         noise_floor=-90.0,
#         bandwidth=20.0e6,
#         frequency=2.4e9,
#         kernel=kernel,
#         parameter_optimization=MCMC(
#             count=10,
#             seed=0,
#             std_params=jnp.asarray([1.0, 1.0, 0.00001], dtype=constant.floating),
#             parameter_optimization=MCMC(
#                 count=1000,
#                 seed=0,
#                 std_params=jnp.asarray([100.0, 100.0, 0.001], dtype=constant.floating),
#                 parameter_optimization=RandomSearch(
#                     count=100000,
#                     seed=0,
#                     lower_bound=jnp.asarray([0.0, 0.0, 0.0], dtype=constant.floating),
#                     upper_bound=jnp.asarray(
#                         [10000.0, 10000.0, 0.1],
#                         dtype=constant.floating,
#                     ),
#                 ),
#             ),
#         ),
#         evaluation_function=evaluation,
#         acquisition_function=acquisition,
#         init_indices_pattern=pattern,
#         init_indices_number=2 if pattern == "grid" else 4,
#         debug_name=debug_name,
#     )
#     log_all_result(
#         debug_name=debug_name,
#         count=count,
#         distance_error=distance_error,
#         data_rate_error=data_rate_error,
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_de.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=distance_error,
#         color_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_dre.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=data_rate_error,
#         color_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_de_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=distance_error,
#         x_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_dre_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=data_rate_error,
#         x_label="Data Rate Error",
#         y_label="Frequency",
#     )


# def double_random_simulation(
#     evaluation: JitWrapped,
#     evaluation_name: str,
#     search_number: int,
# ) -> None:
#     debug_name: str = f"double_random{search_number}_{evaluation_name}"
#     coordinate = Coordinate(
#         x_size=20.0,
#         y_size=20.0,
#         y_mesh=20,
#         x_mesh=20,
#     )
#     propagation = Propagation(
#         seed=0,
#         free_distance=1.0,
#         propagation_coefficient=3.0,
#         distance_correlation=10.0,
#         standard_deviation=8.0,
#     )
#     (
#         count,
#         distance_error,
#         each_distance_error_a,
#         each_distance_error_b,
#         data_rate_error,
#     ) = double_transmitter_random_simulation(
#         coordinate=coordinate,
#         propagation=propagation,
#         simulation_count=1000,
#         receiver_number=5,
#         noise_floor=-90.0,
#         bandwidth=20.0e6,
#         frequency=2.4e9,
#         search_number=search_number,
#         evaluation_function=evaluation,
#         debug_name=debug_name,
#     )
#     log_all_result(
#         debug_name=debug_name,
#         count=count,
#         distance_error=distance_error,
#         data_rate_error=data_rate_error,
#         each_distance_error_a=each_distance_error_a,
#         each_distance_error_b=each_distance_error_b,
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_de.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=distance_error,
#         color_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_dre.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=data_rate_error,
#         color_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_de_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=distance_error,
#         x_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_dre_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=data_rate_error,
#         x_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_scatter_density(
#         file=f"{debug_name}_scatter.pdf",
#         x_value=each_distance_error_a,
#         y_value=each_distance_error_b,
#         color_label="Density",
#     )
#     plot_box(
#         file=f"{debug_name}_box.pdf",
#         data=[
#             count,
#             distance_error,
#             data_rate_error,
#             each_distance_error_a,
#             each_distance_error_b,
#         ],
#         x_tick_labels=[
#             "Count",
#             "Distance Error",
#             "Data Rate Error",
#             "Distance Error A",
#             "Distance Error B",
#         ],
#         y_label="Value",
#     )


# def double_simulation(
#     kernel: Kernel,
#     kernel_name: str,
#     evaluation: JitWrapped,
#     evaluation_name: str,
#     acquisition: JitWrapped,
#     acquisition_name: str,
#     pattern: Literal["grid", "random"],
# ) -> None:
#     debug_name: str = (
#         f"double_{pattern}_{kernel_name}_{evaluation_name}_{acquisition_name}"
#     )
#     coordinate = Coordinate(
#         x_size=20.0,
#         y_size=20.0,
#         y_mesh=20,
#         x_mesh=20,
#     )
#     propagation = Propagation(
#         seed=0,
#         free_distance=1.0,
#         propagation_coefficient=3.0,
#         distance_correlation=10.0,
#         standard_deviation=8.0,
#     )
#     (
#         count,
#         distance_error,
#         each_distance_error_a,
#         each_distance_error_b,
#         data_rate_error,
#     ) = double_transmitter_simulation(
#         coordinate=coordinate,
#         propagation=propagation,
#         simulation_count=1000,
#         receiver_number=5,
#         noise_floor=-90.0,
#         bandwidth=20.0e6,
#         frequency=2.4e9,
#         kernel=kernel,
#         parameter_optimization=MCMC(
#             count=1000,
#             seed=0,
#             std_params=jnp.asarray(
#                 [1.0, 1.0, 1.0, 1.0, 0.00001], dtype=constant.floating
#             ),
#             parameter_optimization=MCMC(
#                 count=1000,
#                 seed=0,
#                 std_params=jnp.asarray(
#                     [100.0, 100.0, 100.0, 100.0, 0.001],
#                     dtype=constant.floating,
#                 ),
#                 parameter_optimization=RandomSearch(
#                     count=100000,
#                     seed=0,
#                     lower_bound=jnp.asarray(
#                         [0.0, 0.0, 0.0, 0.0, 0.0], dtype=constant.floating
#                     ),
#                     upper_bound=jnp.asarray(
#                         [10000.0, 10000.0, 10000.0, 10000.0, 0.1],
#                         dtype=constant.floating,
#                     ),
#                 ),
#             ),
#         ),
#         evaluation_function=evaluation,
#         acquisition_function=acquisition,
#         init_indices_pattern=pattern,
#         init_indices_number=2 if pattern == "grid" else 4,
#         debug_name=debug_name,
#     )
#     log_all_result(
#         debug_name=debug_name,
#         count=count,
#         distance_error=distance_error,
#         data_rate_error=data_rate_error,
#         each_distance_error_a=each_distance_error_a,
#         each_distance_error_b=each_distance_error_b,
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_de.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=distance_error,
#         color_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_dre.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=data_rate_error,
#         color_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_de_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=distance_error,
#         x_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_dre_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=data_rate_error,
#         x_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_scatter_density(
#         file=f"{debug_name}_scatter.pdf",
#         x_value=each_distance_error_a,
#         y_value=each_distance_error_b,
#         color_label="Density",
#     )
#     plot_box(
#         file=f"{debug_name}_box.pdf",
#         data=[
#             count,
#             distance_error,
#             data_rate_error,
#             each_distance_error_a,
#             each_distance_error_b,
#         ],
#         x_tick_labels=[
#             "Count",
#             "Distance Error",
#             "Data Rate Error",
#             "Distance Error A",
#             "Distance Error B",
#         ],
#         y_label="Value",
#     )


# def triple_random_simulation(
#     evaluation: JitWrapped,
#     evaluation_name: str,
#     search_number: int,
# ) -> None:
#     debug_name: str = f"triple_random{search_number}_{evaluation_name}"
#     coordinate = Coordinate(
#         x_size=20.0,
#         y_size=20.0,
#         y_mesh=20,
#         x_mesh=20,
#     )
#     propagation = Propagation(
#         seed=0,
#         free_distance=1.0,
#         propagation_coefficient=3.0,
#         distance_correlation=10.0,
#         standard_deviation=8.0,
#     )
#     (
#         count,
#         distance_error,
#         each_distance_error_a,
#         each_distance_error_b,
#         each_distance_error_c,
#         data_rate_error,
#     ) = triple_transmitter_random_simulation(
#         coordinate=coordinate,
#         propagation=propagation,
#         simulation_count=1000,
#         receiver_number=5,
#         noise_floor=-90.0,
#         bandwidth=20.0e6,
#         frequency=2.4e9,
#         search_number=search_number,
#         evaluation_function=evaluation,
#         debug_name=debug_name,
#     )
#     log_all_result(
#         debug_name=debug_name,
#         count=count,
#         distance_error=distance_error,
#         data_rate_error=data_rate_error,
#         each_distance_error_a=each_distance_error_a,
#         each_distance_error_b=each_distance_error_b,
#         each_distance_error_c=each_distance_error_c,
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_de.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=distance_error,
#         color_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_dre.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=data_rate_error,
#         color_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_de_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=distance_error,
#         x_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_dre_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=data_rate_error,
#         x_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_scatter_density(
#         file=f"{debug_name}_scatter_ab.pdf",
#         x_value=each_distance_error_a,
#         y_value=each_distance_error_b,
#         color_label="Frequency",
#     )
#     plot_scatter_density(
#         file=f"{debug_name}_scatter_bc.pdf",
#         x_value=each_distance_error_b,
#         y_value=each_distance_error_c,
#         color_label="Frequency",
#     )
#     plot_scatter_density(
#         file=f"{debug_name}_scatter_ca.pdf",
#         x_value=each_distance_error_c,
#         y_value=each_distance_error_a,
#         color_label="Frequency",
#     )
#     plot_box(
#         file=f"{debug_name}_box.pdf",
#         data=[
#             count,
#             distance_error,
#             data_rate_error,
#             each_distance_error_a,
#             each_distance_error_b,
#             each_distance_error_c,
#         ],
#         x_tick_labels=[
#             "Count",
#             "Distance Error",
#             "Data Rate Error",
#             "Distance Error A",
#             "Distance Error B",
#             "Distance Error C",
#         ],
#         y_label="Value",
#     )


# def triple_simulation(
#     kernel: Kernel,
#     kernel_name: str,
#     evaluation: JitWrapped,
#     evaluation_name: str,
#     acquisition: JitWrapped,
#     acquisition_name: str,
#     pattern: Literal["grid", "random"],
# ) -> None:
#     debug_name = f"triple_{pattern}_{kernel_name}_{evaluation_name}_{acquisition_name}"
#     coordinate = Coordinate(
#         x_size=20.0,
#         y_size=20.0,
#         y_mesh=20,
#         x_mesh=20,
#     )
#     propagation = Propagation(
#         seed=0,
#         free_distance=1.0,
#         propagation_coefficient=3.0,
#         distance_correlation=10.0,
#         standard_deviation=8.0,
#     )
#     (
#         count,
#         distance_error,
#         each_distance_error_a,
#         each_distance_error_b,
#         each_distance_error_c,
#         data_rate_error,
#     ) = triple_transmitter_simulation(
#         coordinate=coordinate,
#         propagation=propagation,
#         simulation_count=1000,
#         receiver_number=5,
#         noise_floor=-90.0,
#         bandwidth=20.0e6,
#         frequency=2.4e9,
#         kernel=kernel,
#         parameter_optimization=MCMC(
#             count=1000,
#             seed=0,
#             std_params=jnp.asarray(
#                 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.00001], dtype=constant.floating
#             ),
#             parameter_optimization=MCMC(
#                 count=1000,
#                 seed=0,
#                 std_params=jnp.asarray(
#                     [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.001],
#                     dtype=constant.floating,
#                 ),
#                 parameter_optimization=RandomSearch(
#                     count=100000,
#                     seed=0,
#                     lower_bound=jnp.asarray(
#                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=constant.floating
#                     ),
#                     upper_bound=jnp.asarray(
#                         [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 0.1],
#                         dtype=constant.floating,
#                     ),
#                 ),
#             ),
#         ),
#         evaluation_function=evaluation,
#         acquisition_function=acquisition,
#         init_indices_pattern=pattern,
#         init_indices_number=2 if pattern == "grid" else 4,
#         debug_name=debug_name,
#     )
#     log_all_result(
#         debug_name=debug_name,
#         count=count,
#         distance_error=distance_error,
#         data_rate_error=data_rate_error,
#         each_distance_error_a=each_distance_error_a,
#         each_distance_error_b=each_distance_error_b,
#         each_distance_error_c=each_distance_error_c,
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_de.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=distance_error,
#         color_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_heatmap_histogram(
#         file=f"{debug_name}_dre.pdf",
#         horizontal_value=count,
#         x_label="Number of Measurements",
#         color_value=data_rate_error,
#         color_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_de_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=distance_error,
#         x_label="Distance Error",
#         y_label="Frequency",
#     )
#     plot_reverse_heatmap_histogram(
#         file=f"{debug_name}_dre_r.pdf",
#         color_value=count,
#         color_label="Number of Measurements",
#         horizontal_value=data_rate_error,
#         x_label="Data Rate Error",
#         y_label="Frequency",
#     )
#     plot_scatter_density(
#         file=f"{debug_name}_scatter_ab.pdf",
#         x_value=each_distance_error_a,
#         y_value=each_distance_error_b,
#         color_label="Frequency",
#     )
#     plot_scatter_density(
#         file=f"{debug_name}_scatter_bc.pdf",
#         x_value=each_distance_error_b,
#         y_value=each_distance_error_c,
#         color_label="Frequency",
#     )
#     plot_scatter_density(
#         file=f"{debug_name}_scatter_ca.pdf",
#         x_value=each_distance_error_c,
#         y_value=each_distance_error_a,
#         color_label="Frequency",
#     )
#     plot_box(
#         file=f"{debug_name}_box.pdf",
#         data=[
#             count,
#             distance_error,
#             data_rate_error,
#             each_distance_error_a,
#             each_distance_error_b,
#             each_distance_error_c,
#         ],
#         x_tick_labels=[
#             "Count",
#             "Distance Error",
#             "Data Rate Error",
#             "Distance Error A",
#             "Distance Error B",
#             "Distance Error C",
#         ],
#         y_label="Value",
#     )
