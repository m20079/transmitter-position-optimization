import jax
import jax.numpy as jnp
from bayesian_optimization.gaussian_process_regression import (
    GaussianProcessRegression,
)
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.parameter_optimization.parameter_optimization import (
    ParameterOptimization,
)
from constant import floating, integer
from environment.coordinate import Coordinate
from environment.data_rate import DataRate
from environment.distance import get_distance
from environment.propagation import Propagation
from jax import Array
from jax._src.pjit import JitWrapped


def single_transmitter_bayesian_optimization(
    propagation: Propagation,
    coordinate: Coordinate,
    receivers_key: Array,
    shadowing_key: Array,
    receiver_number: int,
    noise_floor: float,
    bandwidth: float,
    x_train_indices: Array,
    y_train_indices: Array,
    kernel: Kernel,
    parameter_optimization: ParameterOptimization,
    evaluation_function: JitWrapped,
    acquisition_function: JitWrapped,
) -> tuple[int, float, float, float]:
    count: int = 0
    min_distance: Array = jnp.asarray(0.0, dtype=floating)
    data_rate_absolute_error: Array = jnp.asarray(0.0, dtype=floating)
    data_rate_relative_error: Array = jnp.asarray(0.0, dtype=floating)

    max_search_number: int = coordinate.get_search_number()

    data_rate: DataRate = propagation.create_data_rate(
        coordinate=coordinate,
        receivers_key=receivers_key,
        shadowing_key=shadowing_key,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
    )
    transmitter_function: JitWrapped = data_rate.create_single_transmitter_function()

    x_transmitter_positions, y_transmitter_positions = (
        coordinate.create_all_transmitter_positions()
    )
    input_test_data: Array = jnp.asarray(
        [
            x_transmitter_positions.ravel(),
            y_transmitter_positions.ravel(),
        ],
        dtype=floating,
    )

    for count in range(x_train_indices.size, max_search_number):
        input_train_data: Array = jnp.asarray(
            coordinate.convert_indices_to_transmitter_positions(
                x_indices=x_train_indices,
                y_indices=y_train_indices,
            )
        )
        output_train_data: Array = evaluation_function(
            data_rate=transmitter_function(
                x_indices=x_train_indices,
                y_indices=y_train_indices,
            ),
        )
        optimized_parameters: Array = parameter_optimization.optimize(
            input_train_data=input_train_data,
            output_train_data=output_train_data,
            kernel=kernel,
        )
        gaussian_process_regression = GaussianProcessRegression(
            input_train_data=input_train_data,
            output_train_data=output_train_data,
            kernel=kernel,
            parameter=optimized_parameters,
        )
        mean, std = gaussian_process_regression.function(
            input_test_data=input_test_data,
        )
        acquisition: Array = acquisition_function(
            mean=mean.reshape(x_transmitter_positions.shape),
            std=std.reshape(x_transmitter_positions.shape),
            max=output_train_data.max(),
            count=count,
        )
        next_index: tuple[Array, ...] = jnp.unravel_index(
            indices=jnp.argmax(acquisition),
            shape=acquisition.shape,
        )
        x_train_indices = jnp.append(x_train_indices, next_index[1])
        y_train_indices = jnp.append(y_train_indices, next_index[0])

        is_finished: Array = jnp.any(
            jnp.all(
                jax.vmap(lambda x: x[:-1] == x[-1])(
                    jnp.asarray(
                        [
                            x_train_indices,
                            y_train_indices,
                        ],
                        dtype=integer,
                    )
                ),
                axis=0,
            )
        )
        if is_finished:
            count: int = count

            max_train_data_index: tuple[Array, ...] = jnp.unravel_index(
                indices=jnp.argmax(output_train_data),
                shape=output_train_data.shape,
            )
            max_x_position, max_y_position = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=x_train_indices[max_train_data_index[0]],
                    y_indices=y_train_indices[max_train_data_index[0]],
                )
            )
            true_x_indices, true_y_indices = (
                data_rate.get_true_single_transmitter_indices(
                    evaluation_function=evaluation_function
                )
            )
            true_x_positions, true_y_positions = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=true_x_indices,
                    y_indices=true_y_indices,
                )
            )
            each_distance: Array = get_distance(
                x_positions_a=true_x_positions,
                y_positions_a=true_y_positions,
                x_positions_b=max_x_position,
                y_positions_b=max_y_position,
            )
            min_distance: Array = each_distance.min()

            true_data_rate: Array = evaluation_function(
                data_rate=transmitter_function(
                    x_indices=true_x_indices,
                    y_indices=true_y_indices,
                )
            )
            data_rate_absolute_error: Array = (
                true_data_rate.max() - output_train_data.max()
            )
            data_rate_relative_error: Array = (
                data_rate_absolute_error / true_data_rate.max()
            )
            break

    return (
        count,
        float(min_distance.block_until_ready()),
        float(data_rate_absolute_error.block_until_ready()),
        float(data_rate_relative_error.block_until_ready()),
    )


def double_transmitter_bayesian_optimization(
    propagation: Propagation,
    coordinate: Coordinate,
    receivers_key: Array,
    shadowing_key: Array,
    receiver_number: int,
    noise_floor: float,
    bandwidth: float,
    x_train_indices_a: Array,
    y_train_indices_a: Array,
    x_train_indices_b: Array,
    y_train_indices_b: Array,
    kernel: Kernel,
    parameter_optimization: ParameterOptimization,
    evaluation_function: JitWrapped,
    acquisition_function: JitWrapped,
) -> tuple[int, float, float, float, float, float]:
    count: int = 0
    min_distance: Array = jnp.asarray(0.0, dtype=floating)
    min_distance_a: Array = jnp.asarray(0.0, dtype=floating)
    min_distance_b: Array = jnp.asarray(0.0, dtype=floating)
    data_rate_absolute_error: Array = jnp.asarray(0.0, dtype=floating)
    data_rate_relative_error: Array = jnp.asarray(0.0, dtype=floating)

    search_number: int = coordinate.get_search_number()
    max_search_number: int = search_number**2

    data_rate: DataRate = propagation.create_data_rate(
        coordinate=coordinate,
        receivers_key=receivers_key,
        shadowing_key=shadowing_key,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
    )
    transmitter_function: JitWrapped = data_rate.create_double_transmitter_function()

    x_transmitter_positions, y_transmitter_positions = (
        coordinate.create_all_transmitter_positions()
    )

    for count in range(x_train_indices_a.size, max_search_number):
        input_train_data_a: Array = jnp.asarray(
            coordinate.convert_indices_to_transmitter_positions(
                x_indices=x_train_indices_a,
                y_indices=y_train_indices_a,
            )
        )
        input_train_data_b: Array = jnp.asarray(
            coordinate.convert_indices_to_transmitter_positions(
                x_indices=x_train_indices_b,
                y_indices=y_train_indices_b,
            )
        )
        input_train_data: Array = jnp.vstack([input_train_data_a, input_train_data_b])
        output_train_data: Array = evaluation_function(
            data_rate=transmitter_function(
                x_indices_a=x_train_indices_a,
                y_indices_a=y_train_indices_a,
                x_indices_b=x_train_indices_b,
                y_indices_b=y_train_indices_b,
            ),
        )
        optimized_parameters: Array = parameter_optimization.optimize(
            input_train_data=input_train_data,
            output_train_data=output_train_data,
            kernel=kernel,
        )
        gaussian_process_regression = GaussianProcessRegression(
            input_train_data=input_train_data,
            output_train_data=output_train_data,
            kernel=kernel,
            parameter=optimized_parameters,
        )

        def body_fun(i: Array) -> tuple[Array, Array]:
            input_x_test_data_b, input_y_test_data_b = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=jnp.full(
                        (x_transmitter_positions.shape), i % (coordinate.x_mesh + 1)
                    ),
                    y_indices=jnp.full(
                        (y_transmitter_positions.shape), i // (coordinate.x_mesh + 1)
                    ),
                )
            )
            mean, std = gaussian_process_regression.function(
                input_test_data=jnp.asarray(
                    [
                        x_transmitter_positions.ravel(),
                        y_transmitter_positions.ravel(),
                        input_x_test_data_b.ravel(),
                        input_y_test_data_b.ravel(),
                    ],
                    dtype=floating,
                ),
            )
            import jax

            acquisition: Array = acquisition_function(
                mean=mean.reshape(x_transmitter_positions.shape),
                std=std.reshape(x_transmitter_positions.shape),
                max=output_train_data.max(),
                count=count,
            )
            jax.debug.print("mean:{}", acquisition)
            next_index: tuple[Array, ...] = jnp.unravel_index(
                indices=jnp.argmax(acquisition),
                shape=acquisition.shape,
            )
            return jnp.asarray(next_index, dtype=integer), acquisition.max()

        next_index, max_acquisition = jax.vmap(body_fun)(jnp.arange(0, search_number))
        max_index: tuple[Array, ...] = jnp.unravel_index(
            indices=jnp.argmax(max_acquisition),
            shape=max_acquisition.shape,
        )
        x_next_index_a: Array = next_index[max_index[0]][1]
        y_next_index_a: Array = next_index[max_index[0]][0]
        x_next_index_b: Array = max_index[0] % (coordinate.x_mesh + 1)
        y_next_index_b: Array = max_index[0] // (coordinate.x_mesh + 1)

        x_train_indices_a = jnp.append(x_train_indices_a, x_next_index_a)
        y_train_indices_a = jnp.append(y_train_indices_a, y_next_index_a)
        x_train_indices_b = jnp.append(x_train_indices_b, x_next_index_b)
        y_train_indices_b = jnp.append(y_train_indices_b, y_next_index_b)

        is_finished: Array = jnp.any(
            jnp.all(
                jax.vmap(lambda x: x[:-1] == x[-1])(
                    jnp.asarray(
                        [
                            x_train_indices_a,
                            y_train_indices_a,
                            x_train_indices_b,
                            y_train_indices_b,
                        ],
                        dtype=integer,
                    ),
                ),
                axis=0,
            )
        )
        if is_finished:
            count: int = count

            max_train_data_index: tuple[Array, ...] = jnp.unravel_index(
                indices=jnp.argmax(output_train_data),
                shape=output_train_data.shape,
            )
            max_x_position_a, max_y_position_a = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=x_train_indices_a[max_train_data_index[0]],
                    y_indices=y_train_indices_a[max_train_data_index[0]],
                )
            )
            max_x_position_b, max_y_position_b = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=x_train_indices_b[max_train_data_index[0]],
                    y_indices=y_train_indices_b[max_train_data_index[0]],
                )
            )
            impure_true_indices: Array = data_rate.get_true_double_transmitter_indices(
                evaluation_function=evaluation_function
            )
            true_indices: Array = impure_true_indices[
                jnp.all(impure_true_indices != -1, axis=1)
            ]
            print(true_indices)
            true_x_positions_a, true_y_positions_a = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=true_indices.T.at[0].get(),
                    y_indices=true_indices.T.at[1].get(),
                )
            )
            true_x_positions_b, true_y_positions_b = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=true_indices.T.at[2].get(),
                    y_indices=true_indices.T.at[3].get(),
                )
            )
            each_distance_a: Array = get_distance(
                x_positions_a=true_x_positions_a,
                y_positions_a=true_y_positions_a,
                x_positions_b=max_x_position_a,
                y_positions_b=max_y_position_a,
            )
            each_distance_b: Array = get_distance(
                x_positions_a=true_x_positions_b,
                y_positions_a=true_y_positions_b,
                x_positions_b=max_x_position_b,
                y_positions_b=max_y_position_b,
            )
            each_distance: Array = jnp.sqrt(each_distance_a**2.0 + each_distance_b**2.0)
            min_distance_index: tuple[Array, ...] = jnp.unravel_index(
                indices=jnp.argmin(each_distance),
                shape=each_distance.shape,
            )
            print(each_distance)
            min_distance_a: Array = each_distance_a[min_distance_index[0]]
            min_distance_b: Array = each_distance_b[min_distance_index[0]]
            min_distance: Array = jnp.sqrt(min_distance_a**2.0 + min_distance_b**2.0)

            true_data_rate: Array = evaluation_function(
                data_rate=transmitter_function(
                    x_indices_a=true_indices.T.at[0].get(),
                    y_indices_a=true_indices.T.at[1].get(),
                    x_indices_b=true_indices.T.at[2].get(),
                    y_indices_b=true_indices.T.at[3].get(),
                )
            )
            data_rate_absolute_error: Array = (
                true_data_rate.max() - output_train_data.max()
            )
            data_rate_relative_error: Array = (
                data_rate_absolute_error / true_data_rate.max()
            )
            break

    return (
        count,
        float(min_distance_a.block_until_ready()),
        float(min_distance_b.block_until_ready()),
        float(min_distance.block_until_ready()),
        float(data_rate_absolute_error.block_until_ready()),
        float(data_rate_relative_error.block_until_ready()),
    )


def triple_transmitter_bayesian_optimization(
    propagation: Propagation,
    coordinate: Coordinate,
    receivers_key: Array,
    shadowing_key: Array,
    receiver_number: int,
    noise_floor: float,
    bandwidth: float,
    x_train_indices_a: Array,
    y_train_indices_a: Array,
    x_train_indices_b: Array,
    y_train_indices_b: Array,
    x_train_indices_c: Array,
    y_train_indices_c: Array,
    kernel: Kernel,
    parameter_optimization: ParameterOptimization,
    evaluation_function: JitWrapped,
    acquisition_function: JitWrapped,
) -> tuple[int, float, float, float, float, float, float]:
    count: int = 0
    min_distance: Array = jnp.asarray(0.0, dtype=floating)
    min_distance_a: Array = jnp.asarray(0.0, dtype=floating)
    min_distance_b: Array = jnp.asarray(0.0, dtype=floating)
    min_distance_c: Array = jnp.asarray(0.0, dtype=floating)
    data_rate_absolute_error: Array = jnp.asarray(0.0, dtype=floating)
    data_rate_relative_error: Array = jnp.asarray(0.0, dtype=floating)

    search_number: int = coordinate.get_search_number()
    max_search_number: int = search_number**3

    data_rate: DataRate = propagation.create_data_rate(
        coordinate=coordinate,
        receivers_key=receivers_key,
        shadowing_key=shadowing_key,
        receiver_number=receiver_number,
        noise_floor=noise_floor,
        bandwidth=bandwidth,
    )
    transmitter_function: JitWrapped = data_rate.create_triple_transmitter_function()

    x_transmitter_positions, y_transmitter_positions = (
        coordinate.create_all_transmitter_positions()
    )

    for count in range(x_train_indices_a.size, max_search_number):
        input_train_data_a: Array = coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_train_indices_a,
            y_indices=y_train_indices_a,
        )
        input_train_data_b: Array = coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_train_indices_b,
            y_indices=y_train_indices_b,
        )
        input_train_data_c: Array = coordinate.convert_indices_to_transmitter_positions(
            x_indices=x_train_indices_c,
            y_indices=y_train_indices_c,
        )
        input_train_data: Array = jnp.vstack(
            [input_train_data_a, input_train_data_b, input_train_data_c]
        )
        output_train_data: Array = evaluation_function(
            data_rate=transmitter_function(
                x_indices_a=x_train_indices_a,
                y_indices_a=y_train_indices_a,
                x_indices_b=x_train_indices_b,
                y_indices_b=y_train_indices_b,
                x_indices_c=x_train_indices_c,
                y_indices_c=y_train_indices_c,
            ),
        )
        optimized_parameters: Array = parameter_optimization.optimize(
            input_train_data=input_train_data,
            output_train_data=output_train_data,
            kernel=kernel,
        )
        gaussian_process_regression = GaussianProcessRegression(
            input_train_data=input_train_data,
            output_train_data=output_train_data,
            kernel=kernel,
            parameter=optimized_parameters,
        )

        def body_fun_j(
            j: Array,
            value: tuple[Array, Array, Array, Array, Array, Array, Array],
        ) -> tuple[Array, Array]:
            def body_fun_i(i: Array) -> tuple[Array, Array]:
                input_x_test_data_b, input_y_test_data_b = (
                    coordinate.convert_indices_to_transmitter_positions(
                        x_indices=jnp.full(
                            (x_transmitter_positions.shape),
                            i % (coordinate.x_mesh + 1),
                        ),
                        y_indices=jnp.full(
                            (y_transmitter_positions.shape),
                            i // (coordinate.x_mesh + 1),
                        ),
                    )
                )
                input_x_test_data_c, input_y_test_data_c = (
                    coordinate.convert_indices_to_transmitter_positions(
                        x_indices=jnp.full(
                            (x_transmitter_positions.shape),
                            j % (coordinate.x_mesh + 1),
                        ),
                        y_indices=jnp.full(
                            (y_transmitter_positions.shape),
                            j // (coordinate.x_mesh + 1),
                        ),
                    )
                )

                mean, std = gaussian_process_regression.function(
                    input_test_data=jnp.asarray(
                        [
                            x_transmitter_positions.ravel(),
                            y_transmitter_positions.ravel(),
                            input_x_test_data_b.ravel(),
                            input_y_test_data_b.ravel(),
                            input_x_test_data_c.ravel(),
                            input_y_test_data_c.ravel(),
                        ],
                        dtype=floating,
                    ),
                )
                acquisition: Array = acquisition_function(
                    mean=mean.reshape(x_transmitter_positions.shape),
                    std=std.reshape(x_transmitter_positions.shape),
                    max=output_train_data.max(),
                    count=count,
                )
                next_index: tuple[Array, ...] = jnp.unravel_index(
                    indices=jnp.argmax(acquisition),
                    shape=acquisition.shape,
                )
                return jnp.asarray(next_index, dtype=integer), acquisition.max()

            next_index, each_max_acquisition = jax.vmap(body_fun_i)(
                jnp.arange(0, search_number)
            )
            max_index: tuple[Array, ...] = jnp.unravel_index(
                indices=jnp.argmax(each_max_acquisition),
                shape=each_max_acquisition.shape,
            )
            x_next_index_a: Array = next_index[max_index[0]][1]
            y_next_index_a: Array = next_index[max_index[0]][0]
            x_next_index_b: Array = max_index[0] % (coordinate.x_mesh + 1)
            y_next_index_b: Array = max_index[0] // (coordinate.x_mesh + 1)
            x_next_index_c: Array = j % (coordinate.x_mesh + 1)
            y_next_index_c: Array = j // (coordinate.x_mesh + 1)

            max_acquisition: Array = each_max_acquisition.max()

            return jax.lax.cond(
                value[0] > max_acquisition,
                lambda: value,
                lambda: (
                    max_acquisition,
                    x_next_index_a,
                    y_next_index_a,
                    x_next_index_b,
                    y_next_index_b,
                    x_next_index_c,
                    y_next_index_c,
                ),
            )

        (
            _,
            x_next_index_a,
            y_next_index_a,
            x_next_index_b,
            y_next_index_b,
            x_next_index_c,
            y_next_index_c,
        ) = jax.lax.fori_loop(
            0,
            search_number,
            body_fun_j,
            (
                jnp.asarray(0.0, dtype=floating),
                jnp.asarray(0, dtype=integer),
                jnp.asarray(0, dtype=integer),
                jnp.asarray(0, dtype=integer),
                jnp.asarray(0, dtype=integer),
                jnp.asarray(0, dtype=integer),
                jnp.asarray(0, dtype=integer),
            ),
        )

        x_train_indices_a = jnp.append(x_train_indices_a, x_next_index_a)
        y_train_indices_a = jnp.append(y_train_indices_a, y_next_index_a)
        x_train_indices_b = jnp.append(x_train_indices_b, x_next_index_b)
        y_train_indices_b = jnp.append(y_train_indices_b, y_next_index_b)
        x_train_indices_c = jnp.append(x_train_indices_c, x_next_index_c)
        y_train_indices_c = jnp.append(y_train_indices_c, y_next_index_c)

        is_finished: Array = jnp.any(
            jnp.all(
                jax.vmap(lambda x: x[:-1] == x[-1])(
                    jnp.asarray(
                        [
                            x_train_indices_a,
                            y_train_indices_a,
                            x_train_indices_b,
                            y_train_indices_b,
                            x_train_indices_c,
                            y_train_indices_c,
                        ],
                        dtype=integer,
                    )
                ),
                axis=0,
            )
        )

        if is_finished:
            count: int = count

            max_train_data_index: tuple[Array, ...] = jnp.unravel_index(
                indices=jnp.argmax(output_train_data),
                shape=output_train_data.shape,
            )
            max_x_position_a, max_y_position_a = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=x_train_indices_a[max_train_data_index[0]],
                    y_indices=y_train_indices_a[max_train_data_index[0]],
                )
            )
            max_x_position_b, max_y_position_b = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=x_train_indices_b[max_train_data_index[0]],
                    y_indices=y_train_indices_b[max_train_data_index[0]],
                )
            )
            max_x_position_c, max_y_position_c = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=x_train_indices_c[max_train_data_index[0]],
                    y_indices=y_train_indices_c[max_train_data_index[0]],
                )
            )
            impure_true_indices = data_rate.get_true_triple_transmitter_indices(
                evaluation_function=evaluation_function
            )
            true_indices: Array = impure_true_indices[
                jnp.all(impure_true_indices != -1, axis=1)
            ]
            true_x_positions_a, true_y_positions_a = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=true_indices.T.at[0].get(),
                    y_indices=true_indices.T.at[1].get(),
                )
            )
            true_x_positions_b, true_y_positions_b = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=true_indices.T.at[2].get(),
                    y_indices=true_indices.T.at[3].get(),
                )
            )
            true_x_positions_c, true_y_positions_c = (
                coordinate.convert_indices_to_transmitter_positions(
                    x_indices=true_indices.T.at[4].get(),
                    y_indices=true_indices.T.at[5].get(),
                )
            )
            each_distance_a: Array = get_distance(
                x_positions_a=true_x_positions_a,
                y_positions_a=true_y_positions_a,
                x_positions_b=max_x_position_a,
                y_positions_b=max_y_position_a,
            )
            each_distance_b: Array = get_distance(
                x_positions_a=true_x_positions_b,
                y_positions_a=true_y_positions_b,
                x_positions_b=max_x_position_b,
                y_positions_b=max_y_position_b,
            )
            each_distance_c: Array = get_distance(
                x_positions_a=true_x_positions_c,
                y_positions_a=true_y_positions_c,
                x_positions_b=max_x_position_c,
                y_positions_b=max_y_position_c,
            )
            each_distance: Array = jnp.sqrt(
                each_distance_a**2.0 + each_distance_b**2.0 + each_distance_c**2.0
            )
            min_distance_index: tuple[Array, ...] = jnp.unravel_index(
                indices=jnp.argmin(each_distance),
                shape=each_distance.shape,
            )
            min_distance_a: Array = each_distance_a[min_distance_index[0]]
            min_distance_b: Array = each_distance_b[min_distance_index[0]]
            min_distance_c: Array = each_distance_c[min_distance_index[0]]
            min_distance: Array = jnp.sqrt(
                min_distance_a**2.0 + min_distance_b**2.0 + min_distance_c**2.0
            )

            true_data_rate: Array = evaluation_function(
                data_rate=transmitter_function(
                    x_indices_a=true_indices.T.at[0].get(),
                    y_indices_a=true_indices.T.at[1].get(),
                    x_indices_b=true_indices.T.at[2].get(),
                    y_indices_b=true_indices.T.at[3].get(),
                    x_indices_c=true_indices.T.at[4].get(),
                    y_indices_c=true_indices.T.at[5].get(),
                )
            )
            data_rate_absolute_error: Array = (
                true_data_rate.max() - output_train_data.max()
            )
            data_rate_relative_error: Array = (
                data_rate_absolute_error / true_data_rate.max()
            )
            break

    return (
        count,
        float(min_distance_a.block_until_ready()),
        float(min_distance_b.block_until_ready()),
        float(min_distance_c.block_until_ready()),
        float(min_distance.block_until_ready()),
        float(data_rate_absolute_error.block_until_ready()),
        float(data_rate_relative_error.block_until_ready()),
    )
