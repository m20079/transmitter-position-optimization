import numpy as np


def print_result(
    debug_name: str,
    count: list[int],
    distance_error: list[float],
    data_rate_absolute_error: list[float],
    data_rate_relative_error: list[float],
    each_distance_error_a: list[float] | None = None,
    each_distance_error_b: list[float] | None = None,
    each_distance_error_c: list[float] | None = None,
) -> None:
    with open(f"{debug_name}.txt", "w") as o:
        print(f"Count: {count}", flush=True, file=o)
        print(f"Distance error: {distance_error}", flush=True, file=o)
        print(
            f"Data Rate absolute error: {data_rate_absolute_error}", flush=True, file=o
        )
        print(
            f"Data Rate relative error: {data_rate_relative_error}", flush=True, file=o
        )

        if each_distance_error_a is not None:
            print(f"Distance error A: {each_distance_error_a}", flush=True, file=o)
        if each_distance_error_b is not None:
            print(f"Distance error B: {each_distance_error_b}", flush=True, file=o)
        if each_distance_error_c is not None:
            print(f"Distance error C: {each_distance_error_c}", flush=True, file=o)

        rounded_distance_error: list[float] = [round(c, 4) for c in distance_error]
        result_distance: dict[float, list[int]] = {}
        for i in sorted(rounded_distance_error):
            if result_distance.get(float(f"{i:0.4f}")) is None:
                result_distance[float(f"{i:0.4f}")] = []

        for i in range(len(count)):
            result_distance[float(f"{rounded_distance_error[i]:0.4f}")].append(count[i])

        print(f"Result Distance: {result_distance}", flush=True, file=o)

        rounded_data_rate_absolute_error: list[float] = [
            round(c, 4) for c in data_rate_absolute_error
        ]
        result_data_rate_absolute: dict[float, list[int]] = {}
        for i in sorted(rounded_data_rate_absolute_error):
            if result_data_rate_absolute.get(float(f"{i:0.4f}")) is None:
                result_data_rate_absolute[float(f"{i:0.4f}")] = []

        for i in range(len(count)):
            result_data_rate_absolute[
                float(f"{rounded_data_rate_absolute_error[i]:0.4f}")
            ].append(count[i])

        rounded_data_rate_relative_error: list[float] = [
            round(c, 4) for c in data_rate_relative_error
        ]
        result_data_rate_relative: dict[float, list[int]] = {}
        for i in sorted(rounded_data_rate_relative_error):
            if result_data_rate_relative.get(float(f"{i:0.4f}")) is None:
                result_data_rate_relative[float(f"{i:0.4f}")] = []

        for i in range(len(count)):
            result_data_rate_relative[
                float(f"{rounded_data_rate_relative_error[i]:0.4f}")
            ].append(count[i])

        print(
            f"Result Data Absolute Rate: {result_data_rate_absolute}",
            flush=True,
            file=o,
        )
        print(
            f"Result Data Relative Rate: {result_data_rate_relative}",
            flush=True,
            file=o,
        )
        print(f"Count average: {np.average(count)}", flush=True, file=o)
        print(f"Distance MAE: {np.average(distance_error)}", flush=True, file=o)
        print(
            f"Data Rate Absolute MAE: {np.average(data_rate_absolute_error)}",
            flush=True,
            file=o,
        )
        print(
            f"Data Rate Relative MAE: {np.average(data_rate_relative_error)}",
            flush=True,
            file=o,
        )
        print(
            f"Distance RMSE: {np.sqrt(np.average(np.square(distance_error)))}",
            flush=True,
            file=o,
        )
        print(
            f"Data Rate Absolute RMSE: {np.sqrt(np.average(np.square(data_rate_absolute_error)))}",
            flush=True,
            file=o,
        )
        print(
            f"Data Rate Relative RMSE: {np.sqrt(np.average(np.square(data_rate_relative_error)))}",
            flush=True,
            file=o,
        )
        print(f"Count STD: {np.std(count)}", flush=True, file=o)
        print(f"Distance STD: {np.std(distance_error)}", flush=True, file=o)
        print(
            f"Data Rate Absolute STD: {np.std(data_rate_absolute_error)}",
            flush=True,
            file=o,
        )
        print(
            f"Data Rate Relative STD: {np.std(data_rate_relative_error)}",
            flush=True,
            file=o,
        )
