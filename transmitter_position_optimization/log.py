import numpy as np


def log_all_result(
    count: list[int],
    distance_error: list[float],
    data_rate_error: list[float],
) -> None:
    print(
        f"Count: {count}",
        flush=True,
    )
    print(
        f"Distance error: {distance_error}",
        flush=True,
    )
    print(
        f"Data Rate error: {data_rate_error}",
        flush=True,
    )

    rounded_distance_error: list[float] = [round(c, 4) for c in distance_error]
    result_distance: dict[float, list[int]] = {}
    for i in sorted(rounded_distance_error):
        if result_distance.get(float(f"{i:0.4f}")) is None:
            result_distance[float(f"{i:0.4f}")] = []

    for i in range(len(count)):
        result_distance[float(f"{rounded_distance_error[i]:0.4f}")].append(count[i])

    print(
        f"Result Distance: {result_distance}",
        flush=True,
    )

    rounded_data_rate_error: list[float] = [round(c, 4) for c in data_rate_error]
    result_data_rate: dict[float, list[int]] = {}
    for i in sorted(rounded_data_rate_error):
        if result_data_rate.get(float(f"{i:0.4f}")) is None:
            result_data_rate[float(f"{i:0.4f}")] = []

    for i in range(len(count)):
        result_data_rate[float(f"{rounded_data_rate_error[i]:0.4f}")].append(count[i])

    print(
        f"Result Data Rate: {result_data_rate}",
        flush=True,
    )
    print(
        f"Count average: {np.average(count)}",
        flush=True,
    )
    print(
        f"Distance MAE: {np.average(distance_error)}",
        flush=True,
    )
    print(
        f"Data Rate MAE: {np.average(data_rate_error)}",
        flush=True,
    )
    print(
        f"Distance RMSE: {np.sqrt(np.average(np.square(distance_error)))}",
        flush=True,
    )
    print(
        f"Data Rate RMSE: {np.sqrt(np.average(np.square(data_rate_error)))}",
        flush=True,
    )
    print(
        f"Count STD: {np.std(count)}",
        flush=True,
    )
    print(
        f"Distance STD: {np.std(distance_error)}",
        flush=True,
    )
    print(
        f"Data Rate STD: {np.std(data_rate_error)}",
        flush=True,
    )
