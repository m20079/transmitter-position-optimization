# from bayesian_optimization.bayesian_optimization import bayesian_optimization
# from environment.coordinate import Coordinate, create_grid_transmitter_indices
# from environment.propagation_space import PropagationSpace
# from environment.receivers import create_random_receivers
# from jax._src.pjit import JitWrapped


# def single_transmitter_simulation(
#     coordinate: Coordinate,
#     propagation_space: PropagationSpace,
#     simulation_count: int,
#     receiver_number: int,
#     noise_floor: float,
#     bandwidth: float,
#     frequency: float,
#     evaluation_method: JitWrapped,
#     evaluation_type: JitWrapped,
#     kernel_function: JitWrapped,
#     parameter_optimization: JitWrapped,
#     acquisition_function: JitWrapped,
# ) -> tuple[list[int], list[float], list[float]]:
#     search_count: list[int] = []
#     distance_error: list[float] = []
#     data_rate_error: list[float] = []

#     for seed in range(simulation_count):
#         x_indices, y_indices = create_grid_transmitter_indices(coordinate, 2)

#         result = bayesian_optimization(
#             coordinate=coordinate,
#             propagation_space=propagation_space.update_seed(seed),
#             frequency=frequency,
#             receivers=create_random_receivers(
#                 coordinate=coordinate,
#                 seed=seed,
#                 number=receiver_number,
#                 noise_floor=noise_floor,
#                 bandwidth=bandwidth,
#             ),
#             evaluation_method=evaluation_method,
#             evaluation_type=evaluation_type,
#             x_train_index=x_indices,
#             y_train_index=y_indices,
#             kernel_function=kernel_function,
#             parameter_optimization=parameter_optimization,
#             acquisition_function=acquisition_function,
#         )
#         search_count.append(result[0])
#         distance_error.append(result[1])
#         data_rate_error.append(result[2])

#     return search_count, distance_error, data_rate_error
