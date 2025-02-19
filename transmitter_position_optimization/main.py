import jax
import jax.numpy as jnp
from constant import floating, integer, platforms
from matplotlib.figure import figaspect
from save import save_result
from simulations import (
    double_transmitter_simulations,
    single_transmitter_simulations,
    triple_transmitter_simulations,
)

if __name__ == "__main__":
    jax.config.update("jax_numpy_dtype_promotion", "strict")
    jax.config.update("jax_platforms", platforms)
    jax.config.update(
        "jax_enable_x64",
        integer == jnp.int64 and floating == jnp.float64,
    )

    # single_transmitter_simulations()
    # double_transmitter_simulations()
    # triple_transmitter_simulations()

    gau = jnp.load("21_gau_times.npz")
    exp = jnp.load("21_exp_times.npz")
    mat3 = jnp.load("21_mat3_times.npz")
    mat5 = jnp.load("21_mat5_times.npz")
    rqp = jnp.load("21_rqp_times.npz")

    name = "21_times"

    # matplotlibで箱ひげ図
    import japanize_matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot(
        [
            gau["distance_error"],
            mat5["distance_error"],
            mat3["distance_error"],
            exp["distance_error"],
            rqp["distance_error"],
        ],
        tick_labels=[
            "Gaussian",
            "Matern5/2",
            "Matern3/2",
            "Exponential",
            "RQ",
        ],
        showmeans=True,
        medianprops=dict(color="red"),
    )
    ax.set_ylabel("真値からの距離 [m]", fontdict={"fontsize": 16})
    ax.tick_params(direction="in")
    fig.savefig(f"{name}_distance_error.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot(
        [
            gau["data_rate_absolute_error"],
            mat5["data_rate_absolute_error"],
            mat3["data_rate_absolute_error"],
            exp["data_rate_absolute_error"],
            rqp["data_rate_absolute_error"],
        ],
        tick_labels=[
            "Gaussian",
            "Matern5/2",
            "Matern3/2",
            "Exponential",
            "RQ",
        ],
        showmeans=True,
        medianprops=dict(color="red"),
    )
    ax.set_ylabel("真値の通信速度との差 [Mbps]", fontdict={"fontsize": 16})
    ax.tick_params(direction="in")
    fig.savefig(f"{name}_data_rate_error.pdf", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot(
        [
            gau["count"],
            mat5["count"],
            mat3["count"],
            exp["count"],
            rqp["count"],
        ],
        tick_labels=[
            "Gaussian",
            "Matern5/2",
            "Matern3/2",
            "Exponential",
            "RQ",
        ],
        showmeans=True,
        medianprops=dict(color="red"),
    )
    ax.set_ylabel("測定回数 [回]", fontdict={"fontsize": 16})
    ax.tick_params(direction="in")
    fig.savefig(f"{name}_count.pdf", bbox_inches="tight")
