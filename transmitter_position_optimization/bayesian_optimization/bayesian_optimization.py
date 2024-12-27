# from functools import partial

# import jax
# import jax.numpy as jnp
# from bayesian_optimization.gaussian_process_regression import (
#     gaussian_process_regression,
# )
# from environment.coordinate import (
#     Coordinate,
#     convert_indices_to_transmitter_positions,
#     create_all_transmitter_positions,
# )
# from environment.propagation_space import PropagationSpace
# from environment.receivers import Receivers
# from environment.transmitter_function import (
#     create_double_transmitter_function,
#     create_single_transmitter_function,
#     create_triple_transmitter_function,
# )
# from environment.utility import convert_positions_to_distance
# from graph import (
#     create_data_rate_heatmap_double,
#     create_data_rate_heatmap_single,
#     create_data_rate_heatmap_triple,
# )
# from jax import Array
# from jax._src.pjit import JitWrapped


# # @partial(jax.jit, static_argnums=(0, 5, 6, 7, 8, 9))
# def get_next_index(
#     coordinate: Coordinate,
#     x_input_train_index: Array,
#     y_input_train_index: Array,
#     x_input_test_data: Array,
#     y_input_test_data: Array,
#     evaluation_type: JitWrapped,
#     single_transmitter_function: JitWrapped,
#     kernel_function: JitWrapped,
#     parameter_optimization: JitWrapped,
#     acquisition_function: JitWrapped,
#     count: int,
# ) -> tuple[tuple[Array, ...], Array]:
#     x_input_train_data, y_input_train_data = convert_indices_to_transmitter_positions(
#         coordinate=coordinate,
#         x_indices=x_input_train_index,
#         y_indices=y_input_train_index,
#     )
#     output_train_data = evaluation_type(
#         single_transmitter_function(x_input_train_index, y_input_train_index), 1
#     )
#     gaussian_process_function = gaussian_process_regression(
#         input_train_data=jnp.array([x_input_train_data, y_input_train_data]),
#         output_train_data=output_train_data,
#         kernel_function=kernel_function,
#         parameter_optimization=parameter_optimization,
#     )
#     mean, sigma = gaussian_process_function(
#         x_test_data=jnp.array(
#             [
#                 x_input_test_data.flatten(),
#                 y_input_test_data.flatten(),
#             ]
#         ),
#         x_test_data_shape=x_input_test_data.shape,
#     )
#     acquisition: Array = acquisition_function(
#         mean,
#         sigma,
#         output_train_data.max(),
#         count,
#     )
#     new_index: tuple[Array, ...] = jnp.unravel_index(
#         indices=jnp.argmax(acquisition),
#         shape=acquisition.shape,
#     )

#     create_data_rate_heatmap_single(
#         file="result.png",
#         data=acquisition,
#         coordinate=coordinate,
#         x_transmitter_positions=x_input_train_data,
#         y_transmitter_positions=y_input_train_data,
#     )

#     return new_index, acquisition.max()


# def bayesian_optimization(
#     coordinate: Coordinate,
#     propagation_space: PropagationSpace,
#     frequency: float,
#     receivers: Receivers,
#     evaluation_method: JitWrapped,
#     evaluation_type: JitWrapped,
#     x_train_index: Array,
#     y_train_index: Array,
#     kernel_function: JitWrapped,
#     parameter_optimization: JitWrapped,
#     acquisition_function: JitWrapped,
# ) -> tuple[int, float, float]:
#     single_transmitter_function, max_evaluation, max_x_indices, max_y_indices = (
#         create_single_transmitter_function(
#             coordinate=coordinate,
#             propagation_space=propagation_space,
#             frequency=frequency,
#             receivers=receivers,
#             evaluation_method=evaluation_method,
#         )
#     )
#     x_test_data, y_test_data = create_all_transmitter_positions(coordinate=coordinate)
#     max_search_number: Array = coordinate.x_mesh * coordinate.y_mesh

#     count = int(max_search_number)
#     distance = jnp.array(jnp.inf)
#     data_rate_error = jnp.array(jnp.inf)

#     for count in range(x_train_index.size, max_search_number):
#         new_index, _ = get_next_index(
#             coordinate=coordinate,
#             x_input_train_index=x_train_index.block_until_ready(),
#             y_input_train_index=y_train_index.block_until_ready(),
#             x_input_test_data=x_test_data.block_until_ready(),
#             y_input_test_data=y_test_data.block_until_ready(),
#             evaluation_type=evaluation_type,
#             single_transmitter_function=single_transmitter_function,
#             kernel_function=kernel_function,
#             parameter_optimization=parameter_optimization,
#             acquisition_function=acquisition_function,
#             count=count,
#         )
#         print(x_train_index, y_train_index, new_index[1], new_index[0])

#         is_next: Array = jnp.any(
#             jnp.all(
#                 jnp.array(
#                     [x_train_index == new_index[1], y_train_index == new_index[0]]
#                 ),
#                 axis=0,
#             )
#         )
#         x_train_index = jnp.append(x_train_index, new_index[1]).block_until_ready()
#         y_train_index = jnp.append(y_train_index, new_index[0]).block_until_ready()

#         if is_next:
#             measured_data = evaluation_type(
#                 single_transmitter_function(x_train_index, y_train_index), 1
#             )
#             max_train_data_index: tuple[Array, ...] = jnp.unravel_index(
#                 jnp.argmax(measured_data),
#                 measured_data.shape,
#             )
#             max_x_position, max_y_position = convert_indices_to_transmitter_positions(
#                 coordinate=coordinate,
#                 x_indices=max_x_indices,
#                 y_indices=max_y_indices,
#             )
#             final_x_position, final_y_position = (
#                 convert_indices_to_transmitter_positions(
#                     coordinate=coordinate,
#                     x_indices=x_train_index[max_train_data_index],
#                     y_indices=y_train_index[max_train_data_index],
#                 )
#             )
#             distance: Array = convert_positions_to_distance(
#                 x_positions_a=max_x_position[0],
#                 y_positions_a=max_y_position[0],
#                 x_positions_b=final_x_position,
#                 y_positions_b=final_y_position,
#             )
#             data_rate_error: Array = jnp.abs(max_evaluation.min - measured_data.max())
#             count: int = count
#             break
#     return count, float(distance), float(data_rate_error)


# # @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 9, 10))
# def get_next_index_2dim(
#     coordinate: Coordinate,
#     x_input_train_index_a: Array,
#     y_input_train_index_a: Array,
#     x_input_train_index_b: Array,
#     y_input_train_index_b: Array,
#     x_input_test_data_a: Array,
#     y_input_test_data_a: Array,
#     x_input_test_data_b: Array,
#     y_input_test_data_b: Array,
#     evaluation_type: JitWrapped,
#     double_transmitter_function: JitWrapped,
#     kernel_function: JitWrapped,
#     parameter_optimization: JitWrapped,
#     acquisition_function: JitWrapped,
#     count: int,
# ) -> tuple[tuple[Array, ...], Array]:
#     x_input_train_data_a, y_input_train_data_a = (
#         convert_indices_to_transmitter_positions(
#             coordinate=coordinate,
#             x_indices=x_input_train_index_a,
#             y_indices=y_input_train_index_a,
#         )
#     )
#     x_input_train_data_b, y_input_train_data_b = (
#         convert_indices_to_transmitter_positions(
#             coordinate=coordinate,
#             x_indices=x_input_train_index_b,
#             y_indices=y_input_train_index_b,
#         )
#     )
#     output_train_data = evaluation_type(
#         double_transmitter_function(
#             x_input_train_index_a,
#             y_input_train_index_a,
#             x_input_train_index_b,
#             y_input_train_index_b,
#         ),
#         1,
#     )
#     gaussian_process_function = gaussian_process_regression(
#         input_train_data=jnp.array(
#             [
#                 x_input_train_data_a,
#                 y_input_train_data_a,
#                 x_input_train_data_b,
#                 y_input_train_data_b,
#             ]
#         ),
#         output_train_data=output_train_data,
#         kernel_function=kernel_function,
#         parameter_optimization=parameter_optimization,
#     )
#     mean, sigma = gaussian_process_function(
#         x_test_data=jnp.array(
#             [
#                 x_input_test_data_a.flatten(),
#                 y_input_test_data_a.flatten(),
#                 x_input_test_data_b.flatten(),
#                 y_input_test_data_b.flatten(),
#             ]
#         ),
#         x_test_data_shape=x_input_test_data_a.shape,
#     )
#     acquisition: Array = acquisition_function(
#         mean,
#         sigma,
#         output_train_data.max(),
#         count,
#     )
#     new_index: tuple[Array, ...] = jnp.unravel_index(
#         indices=jnp.argmax(acquisition),
#         shape=acquisition.shape,
#     )

#     return new_index, acquisition.max()


# def bayesian_optimization_2dim(
#     coordinate: Coordinate,
#     propagation_space: PropagationSpace,
#     frequency: float,
#     receivers: Receivers,
#     evaluation_method: JitWrapped,
#     evaluation_type: JitWrapped,
#     x_train_index_a: Array,
#     y_train_index_a: Array,
#     x_train_index_b: Array,
#     y_train_index_b: Array,
#     kernel_function: JitWrapped,
#     parameter_optimization: JitWrapped,
#     acquisition_function: JitWrapped,
# ) -> tuple[float, float, int]:
#     double_transmitter_function, max_evaluation, max_x_indices, max_y_indices = (
#         create_double_transmitter_function(
#             coordinate=coordinate,
#             propagation_space=propagation_space,
#             frequency=frequency,
#             receivers=receivers,
#             evaluation_method=evaluation_method,
#         )
#     )
#     x_test_data_a, y_test_data_a = create_all_transmitter_positions(
#         coordinate=coordinate
#     )
#     max_search_number: Array = ((coordinate.x_mesh + 1) * (coordinate.y_mesh + 1)) ** 2
#     loop_number: Array = (coordinate.x_mesh + 1) * (coordinate.y_mesh + 1)

#     count = int(max_search_number)
#     distance = jnp.array(jnp.inf)
#     data_rate_error = jnp.array(jnp.inf)

#     for count in range(x_train_index_a.size, max_search_number):

#         def body_fun(
#             i: Array,
#         ):
#             x_test_data_b, y_test_data_b = convert_indices_to_transmitter_positions(
#                 coordinate=coordinate,
#                 x_indices=jnp.full((x_test_data_a.shape), i % coordinate.x_mesh),
#                 y_indices=jnp.full((y_test_data_a.shape), i // coordinate.x_mesh),
#             )
#             return get_next_index_2dim(
#                 coordinate=coordinate,
#                 x_input_train_index_a=x_train_index_a.block_until_ready(),
#                 y_input_train_index_a=y_train_index_a.block_until_ready(),
#                 x_input_train_index_b=x_train_index_b.block_until_ready(),
#                 y_input_train_index_b=y_train_index_b.block_until_ready(),
#                 x_input_test_data_a=x_test_data_a.block_until_ready(),
#                 y_input_test_data_a=y_test_data_a.block_until_ready(),
#                 x_input_test_data_b=x_test_data_b,
#                 y_input_test_data_b=y_test_data_b,
#                 evaluation_type=evaluation_type,
#                 double_transmitter_function=double_transmitter_function,
#                 kernel_function=kernel_function,
#                 parameter_optimization=parameter_optimization,
#                 acquisition_function=acquisition_function,
#                 count=count,
#             )

#         next_index = jax.vmap(body_fun)(jnp.arange(0, loop_number))
#         max_index = jnp.unravel_index(
#             indices=jnp.argmax(next_index[1]),
#             shape=next_index[1].shape,
#         )
#         new_index_a = jnp.array(
#             [next_index[0][0][max_index[0]], next_index[0][1][max_index[0]]]
#         )
#         new_index_b = jnp.array(
#             [max_index[0] // coordinate.x_mesh, max_index[0] % coordinate.x_mesh]
#         )

#         create_data_rate_heatmap_double(
#             file="result.png",
#             data=jnp.array([[0]]),
#             coordinate=coordinate,
#             x_transmitter_positions_a=x_train_index_a,
#             y_transmitter_positions_a=y_train_index_a,
#             x_transmitter_positions_b=x_train_index_b,
#             y_transmitter_positions_b=y_train_index_b,
#         )

#         is_next: Array = jnp.logical_and(
#             jnp.any(
#                 jnp.all(
#                     jnp.array(
#                         [
#                             x_train_index_a == new_index_a[1],
#                             y_train_index_a == new_index_a[0],
#                         ]
#                     ),
#                     axis=0,
#                 )
#             ),
#             jnp.any(
#                 jnp.all(
#                     jnp.array(
#                         [
#                             x_train_index_b == new_index_b[1],
#                             y_train_index_b == new_index_b[0],
#                         ]
#                     ),
#                     axis=0,
#                 )
#             ),
#         )
#         x_train_index_a = jnp.append(x_train_index_a, new_index_a[1])
#         y_train_index_a = jnp.append(y_train_index_a, new_index_a[0])
#         x_train_index_b = jnp.append(x_train_index_b, new_index_b[1])
#         y_train_index_b = jnp.append(y_train_index_b, new_index_b[0])

#         print(x_train_index_a, y_train_index_a, x_train_index_b, y_train_index_b)

#         if is_next:
#             measured_data = evaluation_type(
#                 double_transmitter_function(
#                     x_train_index_a,
#                     y_train_index_a,
#                     x_train_index_b,
#                     y_train_index_b,
#                 ),
#                 1,
#             )

#             max_train_data_index: tuple[Array, ...] = jnp.unravel_index(
#                 jnp.argmax(measured_data),
#                 measured_data.shape,
#             )
#             max_x_position, max_y_position = convert_indices_to_transmitter_positions(
#                 coordinate=coordinate,
#                 x_indices=max_x_indices,
#                 y_indices=max_y_indices,
#             )
#             final_x_position_a, final_y_position_a = (
#                 convert_indices_to_transmitter_positions(
#                     coordinate=coordinate,
#                     x_indices=x_train_index_a[max_train_data_index],
#                     y_indices=y_train_index_a[max_train_data_index],
#                 )
#             )
#             final_x_position_b, final_y_position_b = (
#                 convert_indices_to_transmitter_positions(
#                     coordinate=coordinate,
#                     x_indices=x_train_index_b[max_train_data_index],
#                     y_indices=y_train_index_b[max_train_data_index],
#                 )
#             )
#             distance_a_b: Array = jnp.sqrt(
#                 jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[0],
#                         y_positions_a=max_y_position[0],
#                         x_positions_b=final_x_position_a,
#                         y_positions_b=final_y_position_a,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[1],
#                         y_positions_a=max_y_position[1],
#                         x_positions_b=final_x_position_b,
#                         y_positions_b=final_y_position_b,
#                     ),
#                     2,
#                 )
#             )
#             distance_b_a: Array = jnp.sqrt(
#                 jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[0],
#                         y_positions_a=max_y_position[0],
#                         x_positions_b=final_x_position_b,
#                         y_positions_b=final_y_position_b,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[1],
#                         y_positions_a=max_y_position[1],
#                         x_positions_b=final_x_position_a,
#                         y_positions_b=final_y_position_a,
#                     ),
#                     2,
#                 )
#             )
#             distance: Array = jnp.minimum(distance_a_b, distance_b_a)
#             data_rate_error: Array = jnp.abs(max_evaluation.min - measured_data.max())
#             count: int = count
#             break
#     return float(distance), float(data_rate_error), count


# # @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 9, 10))
# def get_next_index_3dim(
#     coordinate: Coordinate,
#     x_input_train_index_a: Array,
#     y_input_train_index_a: Array,
#     x_input_train_index_b: Array,
#     y_input_train_index_b: Array,
#     x_input_train_index_c: Array,
#     y_input_train_index_c: Array,
#     x_input_test_data_a: Array,
#     y_input_test_data_a: Array,
#     x_input_test_data_b: Array,
#     y_input_test_data_b: Array,
#     x_input_test_data_c: Array,
#     y_input_test_data_c: Array,
#     evaluation_type: JitWrapped,
#     triple_transmitter_function: JitWrapped,
#     kernel_function: JitWrapped,
#     parameter_optimization: JitWrapped,
#     acquisition_function: JitWrapped,
#     count: int,
# ) -> tuple[tuple[Array, ...], Array]:
#     x_input_train_data_a, y_input_train_data_a = (
#         convert_indices_to_transmitter_positions(
#             coordinate=coordinate,
#             x_indices=x_input_train_index_a,
#             y_indices=y_input_train_index_a,
#         )
#     )
#     x_input_train_data_b, y_input_train_data_b = (
#         convert_indices_to_transmitter_positions(
#             coordinate=coordinate,
#             x_indices=x_input_train_index_b,
#             y_indices=y_input_train_index_b,
#         )
#     )
#     x_input_train_data_c, y_input_train_data_c = (
#         convert_indices_to_transmitter_positions(
#             coordinate=coordinate,
#             x_indices=x_input_train_index_c,
#             y_indices=y_input_train_index_c,
#         )
#     )
#     output_train_data = evaluation_type(
#         triple_transmitter_function(
#             x_input_train_index_a,
#             y_input_train_index_a,
#             x_input_train_index_b,
#             y_input_train_index_b,
#             x_input_train_index_c,
#             y_input_train_index_c,
#         ),
#         1,
#     )
#     gaussian_process_function = gaussian_process_regression(
#         input_train_data=jnp.array(
#             [
#                 x_input_train_data_a,
#                 y_input_train_data_a,
#                 x_input_train_data_b,
#                 y_input_train_data_b,
#                 x_input_train_data_c,
#                 y_input_train_data_c,
#             ]
#         ),
#         output_train_data=output_train_data,
#         kernel_function=kernel_function,
#         parameter_optimization=parameter_optimization,
#     )
#     mean, sigma = gaussian_process_function(
#         x_test_data=jnp.array(
#             [
#                 x_input_test_data_a.flatten(),
#                 y_input_test_data_a.flatten(),
#                 x_input_test_data_b.flatten(),
#                 y_input_test_data_b.flatten(),
#                 x_input_test_data_c.flatten(),
#                 y_input_test_data_c.flatten(),
#             ]
#         ),
#         x_test_data_shape=x_input_test_data_a.shape,
#     )
#     acquisition: Array = acquisition_function(
#         mean,
#         sigma,
#         output_train_data.max(),
#         count,
#     )
#     new_index: tuple[Array, ...] = jnp.unravel_index(
#         indices=jnp.argmax(acquisition),
#         shape=acquisition.shape,
#     )

#     return new_index, acquisition.max()


# def bayesian_optimization_3dim(
#     coordinate: Coordinate,
#     propagation_space: PropagationSpace,
#     frequency: float,
#     receivers: Receivers,
#     evaluation_method: JitWrapped,
#     evaluation_type: JitWrapped,
#     x_train_index_a: Array,
#     y_train_index_a: Array,
#     x_train_index_b: Array,
#     y_train_index_b: Array,
#     x_train_index_c: Array,
#     y_train_index_c: Array,
#     kernel_function: JitWrapped,
#     parameter_optimization: JitWrapped,
#     acquisition_function: JitWrapped,
# ) -> tuple[float, float, int]:
#     triple_transmitter_function, max_evaluation, max_x_indices, max_y_indices = (
#         create_triple_transmitter_function(
#             coordinate=coordinate,
#             propagation_space=propagation_space,
#             frequency=frequency,
#             receivers=receivers,
#             evaluation_method=evaluation_method,
#         )
#     )
#     x_test_data_a, y_test_data_a = create_all_transmitter_positions(
#         coordinate=coordinate
#     )
#     max_search_number: Array = (coordinate.x_mesh * coordinate.y_mesh) ** 2
#     loop_number: Array = coordinate.x_mesh * coordinate.y_mesh

#     count = int(max_search_number)
#     distance = jnp.array(jnp.inf)
#     data_rate_error = jnp.array(jnp.inf)

#     for count in range(x_train_index_a.size, max_search_number):

#         def body_fun1(
#             i: Array,
#             value: tuple[
#                 tuple[Array, Array], tuple[Array, Array], tuple[Array, Array], Array
#             ],
#         ) -> tuple[
#             tuple[Array, Array], tuple[Array, Array], tuple[Array, Array], Array
#         ]:
#             def body_fun2(
#                 j: Array,
#                 value2: tuple[
#                     tuple[Array, Array], tuple[Array, Array], tuple[Array, Array], Array
#                 ],
#             ) -> tuple[
#                 tuple[Array, Array], tuple[Array, Array], tuple[Array, Array], Array
#             ]:
#                 _, _, _, current_max_acquisition = value2

#                 x_test_data_b, y_test_data_b = convert_indices_to_transmitter_positions(
#                     coordinate=coordinate,
#                     x_indices=jnp.full((x_test_data_a.shape), i % coordinate.x_mesh),
#                     y_indices=jnp.full((y_test_data_a.shape), i // coordinate.x_mesh),
#                 )
#                 x_test_data_c, y_test_data_c = convert_indices_to_transmitter_positions(
#                     coordinate=coordinate,
#                     x_indices=jnp.full((x_test_data_a.shape), j % coordinate.x_mesh),
#                     y_indices=jnp.full((y_test_data_a.shape), j // coordinate.x_mesh),
#                 )
#                 new_index, max_acquisition = get_next_index_3dim(
#                     coordinate=coordinate,
#                     x_input_train_index_a=x_train_index_a.block_until_ready(),
#                     y_input_train_index_a=y_train_index_a.block_until_ready(),
#                     x_input_train_index_b=x_train_index_b.block_until_ready(),
#                     y_input_train_index_b=y_train_index_b.block_until_ready(),
#                     x_input_train_index_c=x_train_index_c.block_until_ready(),
#                     y_input_train_index_c=y_train_index_c.block_until_ready(),
#                     x_input_test_data_a=x_test_data_a.block_until_ready(),
#                     y_input_test_data_a=y_test_data_a.block_until_ready(),
#                     x_input_test_data_b=x_test_data_b,
#                     y_input_test_data_b=y_test_data_b,
#                     x_input_test_data_c=x_test_data_c,
#                     y_input_test_data_c=y_test_data_c,
#                     evaluation_type=evaluation_type,
#                     triple_transmitter_function=triple_transmitter_function,
#                     kernel_function=kernel_function,
#                     parameter_optimization=parameter_optimization,
#                     acquisition_function=acquisition_function,
#                     count=count,
#                 )
#                 return jax.lax.cond(
#                     current_max_acquisition > max_acquisition,
#                     lambda: value2,
#                     lambda: (
#                         new_index,
#                         (i // coordinate.x_mesh, i % coordinate.x_mesh),
#                         (j // coordinate.x_mesh, j % coordinate.x_mesh),
#                         max_acquisition,
#                     ),
#                 )

#             return jax.lax.fori_loop(0, loop_number, body_fun2, value)

#         new_index_a, new_index_b, new_index_c, _ = jax.lax.fori_loop(
#             0,
#             loop_number,
#             body_fun1,
#             (
#                 (jnp.array(0), jnp.array(0)),
#                 (jnp.array(0), jnp.array(0)),
#                 (jnp.array(0), jnp.array(0)),
#                 jnp.array(-jnp.inf),
#             ),
#         )

#         create_data_rate_heatmap_triple(
#             file="result.png",
#             data=jnp.array([[0]]),
#             coordinate=coordinate,
#             x_transmitter_positions_a=x_train_index_a,
#             y_transmitter_positions_a=y_train_index_a,
#             x_transmitter_positions_b=x_train_index_b,
#             y_transmitter_positions_b=y_train_index_b,
#             x_transmitter_positions_c=x_train_index_c,
#             y_transmitter_positions_c=y_train_index_c,
#         )

#         is_next: Array = jnp.logical_and(
#             jnp.any(
#                 jnp.all(
#                     jnp.array(
#                         [
#                             x_train_index_a == new_index_a[1],
#                             y_train_index_a == new_index_a[0],
#                         ]
#                     ),
#                     axis=0,
#                 )
#             ),
#             jnp.logical_and(
#                 jnp.any(
#                     jnp.all(
#                         jnp.array(
#                             [
#                                 x_train_index_b == new_index_b[1],
#                                 y_train_index_b == new_index_b[0],
#                             ]
#                         ),
#                         axis=0,
#                     )
#                 ),
#                 jnp.any(
#                     jnp.all(
#                         jnp.array(
#                             [
#                                 x_train_index_c == new_index_c[1],
#                                 y_train_index_c == new_index_c[0],
#                             ]
#                         ),
#                         axis=0,
#                     )
#                 ),
#             ),
#         )
#         x_train_index_a = jnp.append(x_train_index_a, new_index_a[1])
#         y_train_index_a = jnp.append(y_train_index_a, new_index_a[0])
#         x_train_index_b = jnp.append(x_train_index_b, new_index_b[1])
#         y_train_index_b = jnp.append(y_train_index_b, new_index_b[0])
#         x_train_index_c = jnp.append(x_train_index_c, new_index_c[1])
#         y_train_index_c = jnp.append(y_train_index_c, new_index_c[0])

#         print(x_train_index_a, y_train_index_a, x_train_index_b, y_train_index_b)

#         if is_next:
#             measured_data = evaluation_type(
#                 triple_transmitter_function(
#                     x_train_index_a,
#                     y_train_index_a,
#                     x_train_index_b,
#                     y_train_index_b,
#                     x_train_index_c,
#                     y_train_index_c,
#                 ),
#                 1,
#             )

#             max_train_data_index: tuple[Array, ...] = jnp.unravel_index(
#                 jnp.argmax(measured_data),
#                 measured_data.shape,
#             )
#             max_x_position, max_y_position = convert_indices_to_transmitter_positions(
#                 coordinate=coordinate,
#                 x_indices=max_x_indices,
#                 y_indices=max_y_indices,
#             )
#             final_x_position_a, final_y_position_a = (
#                 convert_indices_to_transmitter_positions(
#                     coordinate=coordinate,
#                     x_indices=x_train_index_a[max_train_data_index],
#                     y_indices=y_train_index_a[max_train_data_index],
#                 )
#             )
#             final_x_position_b, final_y_position_b = (
#                 convert_indices_to_transmitter_positions(
#                     coordinate=coordinate,
#                     x_indices=x_train_index_b[max_train_data_index],
#                     y_indices=y_train_index_b[max_train_data_index],
#                 )
#             )
#             final_x_position_c, final_y_position_c = (
#                 convert_indices_to_transmitter_positions(
#                     coordinate=coordinate,
#                     x_indices=x_train_index_c[max_train_data_index],
#                     y_indices=y_train_index_c[max_train_data_index],
#                 )
#             )
#             distance_a_b_c: Array = jnp.sqrt(
#                 jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[0],
#                         y_positions_a=max_y_position[0],
#                         x_positions_b=final_x_position_a,
#                         y_positions_b=final_y_position_a,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[1],
#                         y_positions_a=max_y_position[1],
#                         x_positions_b=final_x_position_b,
#                         y_positions_b=final_y_position_b,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[2],
#                         y_positions_a=max_y_position[2],
#                         x_positions_b=final_x_position_c,
#                         y_positions_b=final_y_position_c,
#                     ),
#                     2,
#                 )
#             )
#             distance_a_c_b: Array = jnp.sqrt(
#                 jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[0],
#                         y_positions_a=max_y_position[0],
#                         x_positions_b=final_x_position_a,
#                         y_positions_b=final_y_position_a,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[1],
#                         y_positions_a=max_y_position[1],
#                         x_positions_b=final_x_position_c,
#                         y_positions_b=final_y_position_c,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[2],
#                         y_positions_a=max_y_position[2],
#                         x_positions_b=final_x_position_b,
#                         y_positions_b=final_y_position_b,
#                     ),
#                     2,
#                 )
#             )
#             distance_b_a_c: Array = jnp.sqrt(
#                 jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[0],
#                         y_positions_a=max_y_position[0],
#                         x_positions_b=final_x_position_b,
#                         y_positions_b=final_y_position_b,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[1],
#                         y_positions_a=max_y_position[1],
#                         x_positions_b=final_x_position_a,
#                         y_positions_b=final_y_position_a,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[2],
#                         y_positions_a=max_y_position[2],
#                         x_positions_b=final_x_position_c,
#                         y_positions_b=final_y_position_c,
#                     ),
#                     2,
#                 )
#             )
#             distance_c_a_b: Array = jnp.sqrt(
#                 jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[0],
#                         y_positions_a=max_y_position[0],
#                         x_positions_b=final_x_position_c,
#                         y_positions_b=final_y_position_c,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[1],
#                         y_positions_a=max_y_position[1],
#                         x_positions_b=final_x_position_a,
#                         y_positions_b=final_y_position_a,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[2],
#                         y_positions_a=max_y_position[2],
#                         x_positions_b=final_x_position_b,
#                         y_positions_b=final_y_position_b,
#                     ),
#                     2,
#                 )
#             )
#             distance_b_c_a: Array = jnp.sqrt(
#                 jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[0],
#                         y_positions_a=max_y_position[0],
#                         x_positions_b=final_x_position_b,
#                         y_positions_b=final_y_position_b,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[1],
#                         y_positions_a=max_y_position[1],
#                         x_positions_b=final_x_position_c,
#                         y_positions_b=final_y_position_c,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[2],
#                         y_positions_a=max_y_position[2],
#                         x_positions_b=final_x_position_a,
#                         y_positions_b=final_y_position_a,
#                     ),
#                     2,
#                 )
#             )
#             distance_c_b_a: Array = jnp.sqrt(
#                 jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[0],
#                         y_positions_a=max_y_position[0],
#                         x_positions_b=final_x_position_c,
#                         y_positions_b=final_y_position_c,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[1],
#                         y_positions_a=max_y_position[1],
#                         x_positions_b=final_x_position_b,
#                         y_positions_b=final_y_position_b,
#                     ),
#                     2,
#                 )
#                 + jnp.power(
#                     convert_positions_to_distance(
#                         x_positions_a=max_x_position[2],
#                         y_positions_a=max_y_position[2],
#                         x_positions_b=final_x_position_a,
#                         y_positions_b=final_y_position_a,
#                     ),
#                     2,
#                 )
#             )
#             distance: Array = jnp.array(
#                 [
#                     distance_a_b_c,
#                     distance_a_c_b,
#                     distance_b_a_c,
#                     distance_c_a_b,
#                     distance_b_c_a,
#                     distance_c_b_a,
#                 ]
#             ).min()
#             data_rate_error: Array = jnp.abs(max_evaluation.min - measured_data.max())
#             count: int = count
#             break
#     return float(distance), float(data_rate_error), count
