import japanize_matplotlib  # noqa: F401
import matplotlib
import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.pyplot as plt
from environment.coordinate import Coordinate
from environment.receivers import Receivers
from environment.transmitter import Transmitter
from jax import Array
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import MultipleLocator


def plot_rssi_heatmap(
    file: str,
    data: Array,
    coordinate: Coordinate,
    vmin: float | None = None,
    vmax: float | None = None,
    transmitter: Transmitter | None = None,
    receivers: Receivers | None = None,
    transparent: bool = True,
) -> None:
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()
    im: AxesImage = ax.imshow(
        data,
        cmap="jet",
        extent=coordinate.get_receivers_extent(),
        vmin=vmin,
        vmax=vmax,
    )
    if transmitter is not None:
        ax.scatter(
            transmitter.x_position,
            transmitter.y_position,
            color="white",
        )
    if receivers is not None:
        for i in range(receivers.noise_floor.size):
            ax.scatter(
                receivers.x_positions[i],
                receivers.y_positions[i],
                color="black",
            )
    ax.set_xlabel("x [m]", fontsize=15)
    ax.set_ylabel("y [m]", fontsize=15)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.invert_yaxis()
    cb: Colorbar = fig.colorbar(im)
    cb.set_label("RSSI [dBm]", size=15)
    fig.savefig(f"{file}", transparent=transparent, bbox_inches="tight")
    plt.cla()
    plt.close()


def plot_data_rate_heatmap_single(
    file: str,
    data: Array,
    coordinate: Coordinate,
    vmin: float | None = None,
    vmax: float | None = None,
    x_transmitter_positions: Array | None = None,
    y_transmitter_positions: Array | None = None,
    receivers: Receivers | None = None,
    max_x_transmitter_position: Array | None = None,
    max_y_transmitter_position: Array | None = None,
    transparent: bool = True,
) -> None:
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()
    im: AxesImage = ax.imshow(
        data,
        cmap="cool",
        extent=coordinate.get_transmitter_extent(),
        vmin=vmin,
        vmax=vmax,
    )
    if x_transmitter_positions is not None and y_transmitter_positions is not None:
        for i in range(x_transmitter_positions.size):
            ax.text(
                float(x_transmitter_positions[i]),
                float(y_transmitter_positions[i]),
                f"{i+1}",
                color="white",
                ha="center",
                va="center_baseline",
            )
    if receivers is not None:
        for i in range(receivers.noise_floor.size):
            ax.scatter(
                receivers.x_positions[i],
                receivers.y_positions[i],
                color="black",
            )
    if (
        max_x_transmitter_position is not None
        and max_y_transmitter_position is not None
    ):
        ax.scatter(
            max_x_transmitter_position,
            max_y_transmitter_position,
            color="red",
        )
    ax.set_xlabel("x [m]", fontsize=15)
    ax.set_ylabel("y [m]", fontsize=15)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.invert_yaxis()
    cb: Colorbar = fig.colorbar(im)
    cb.set_label("Data Rate [Mbps]", size=15)
    fig.savefig(f"{file}", transparent=transparent, bbox_inches="tight")
    plt.cla()
    plt.close()


def plot_data_rate_heatmap_double(
    file: str,
    data: Array,
    coordinate: Coordinate,
    vmin: float | None = None,
    vmax: float | None = None,
    x_transmitter_positions_a: Array | None = None,
    y_transmitter_positions_a: Array | None = None,
    x_transmitter_positions_b: Array | None = None,
    y_transmitter_positions_b: Array | None = None,
    receivers: Receivers | None = None,
    max_x_transmitter_position_a: Array | None = None,
    max_y_transmitter_position_a: Array | None = None,
    max_x_transmitter_position_b: Array | None = None,
    max_y_transmitter_position_b: Array | None = None,
    transparent: bool = True,
) -> None:
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()
    im: AxesImage = ax.imshow(
        data,
        cmap="cool",
        extent=coordinate.get_transmitter_extent(),
        vmin=vmin,
        vmax=vmax,
    )
    if x_transmitter_positions_a is not None and y_transmitter_positions_a is not None:
        for i in range(x_transmitter_positions_a.size):
            ax.text(
                float(x_transmitter_positions_a[i]),
                float(y_transmitter_positions_a[i]),
                f"{i+1}",
                color="white",
                alpha=0.5,
                ha="center",
                va="center_baseline",
            )
    if x_transmitter_positions_b is not None and y_transmitter_positions_b is not None:
        for i in range(x_transmitter_positions_b.size):
            ax.scatter(
                float(x_transmitter_positions_b[i]),
                float(y_transmitter_positions_b[i]),
                f"{i+1}",
                color="green",
                alpha=0.5,
                ha="center",
                va="center_baseline",
            )
    if receivers is not None:
        for i in range(receivers.noise_floor.size):
            ax.scatter(
                receivers.x_positions[i],
                receivers.y_positions[i],
                color="black",
            )
    if (
        max_x_transmitter_position_a is not None
        and max_y_transmitter_position_a is not None
        and max_x_transmitter_position_b is not None
        and max_y_transmitter_position_b is not None
    ):
        ax.scatter(
            max_x_transmitter_position_a,
            max_y_transmitter_position_a,
            color="red",
        )
        ax.scatter(
            max_x_transmitter_position_b,
            max_y_transmitter_position_b,
            color="red",
        )
    ax.set_xlabel("x [m]", fontsize=15)
    ax.set_ylabel("y [m]", fontsize=15)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.invert_yaxis()
    cb: Colorbar = fig.colorbar(im)
    cb.set_label("Data Rate [Mbps]", size=15)
    fig.savefig(f"{file}", transparent=transparent, bbox_inches="tight")
    plt.cla()
    plt.close()


def plot_data_rate_heatmap_triple(
    file: str,
    data: Array,
    coordinate: Coordinate,
    vmin: float | None = None,
    vmax: float | None = None,
    x_transmitter_positions_a: Array | None = None,
    y_transmitter_positions_a: Array | None = None,
    x_transmitter_positions_b: Array | None = None,
    y_transmitter_positions_b: Array | None = None,
    x_transmitter_positions_c: Array | None = None,
    y_transmitter_positions_c: Array | None = None,
    receivers: Receivers | None = None,
    max_x_transmitter_position_a: Array | None = None,
    max_y_transmitter_position_a: Array | None = None,
    max_x_transmitter_position_b: Array | None = None,
    max_y_transmitter_position_b: Array | None = None,
    max_x_transmitter_position_c: Array | None = None,
    max_y_transmitter_position_c: Array | None = None,
    transparent: bool = True,
) -> None:
    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()
    im: AxesImage = ax.imshow(
        data,
        cmap="cool",
        extent=coordinate.get_transmitter_extent(),
        vmin=vmin,
        vmax=vmax,
    )
    if x_transmitter_positions_a is not None and y_transmitter_positions_a is not None:
        for i in range(x_transmitter_positions_a.size):
            ax.text(
                float(x_transmitter_positions_a[i]),
                float(y_transmitter_positions_a[i]),
                f"{i+1}",
                color="white",
                alpha=0.5,
                ha="center",
                va="center_baseline",
            )
    if x_transmitter_positions_b is not None and y_transmitter_positions_b is not None:
        for i in range(x_transmitter_positions_b.size):
            ax.text(
                float(x_transmitter_positions_b[i]),
                float(y_transmitter_positions_b[i]),
                f"{i+1}",
                color="green",
                alpha=0.5,
                ha="center",
                va="center_baseline",
            )
    if x_transmitter_positions_c is not None and y_transmitter_positions_c is not None:
        for i in range(x_transmitter_positions_c.size):
            ax.text(
                float(x_transmitter_positions_c[i]),
                float(y_transmitter_positions_c[i]),
                f"{i+1}",
                color="blue",
                alpha=0.5,
                ha="center",
                va="center_baseline",
            )
    if receivers is not None:
        for i in range(receivers.noise_floor.size):
            ax.scatter(
                receivers.x_positions[i],
                receivers.y_positions[i],
                color="black",
            )
    if (
        max_x_transmitter_position_a is not None
        and max_y_transmitter_position_a is not None
        and max_x_transmitter_position_b is not None
        and max_y_transmitter_position_b is not None
        and max_x_transmitter_position_c is not None
        and max_y_transmitter_position_c is not None
    ):
        ax.scatter(
            max_x_transmitter_position_a,
            max_y_transmitter_position_a,
            color="red",
        )
        ax.scatter(
            max_x_transmitter_position_b,
            max_y_transmitter_position_b,
            color="green",
        )
        ax.scatter(
            max_x_transmitter_position_c,
            max_y_transmitter_position_c,
            color="blue",
        )
    ax.set_xlabel("x [m]", fontsize=15)
    ax.set_ylabel("y [m]", fontsize=15)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.invert_yaxis()
    cb: Colorbar = fig.colorbar(im)
    cb.set_label("Data Rate [Mbps]", size=15)
    fig.savefig(f"{file}", transparent=transparent, bbox_inches="tight")
    plt.cla()
    plt.close()


def plot_heatmap_histogram(
    file: str,
    horizontal_value: list[int],
    color_value: list[float],
    color_label: str,
    x_label: str,
    y_label: str,
    transparent: bool = True,
) -> None:
    cmap: Colormap = matplotlib.colormaps.get_cmap("jet")
    rounded_color_value: list[float] = [round(c, 4) for c in color_value]
    color: list = []
    result: dict[float, list[int]] = {}
    for i in sorted(rounded_color_value):
        if result.get(float(f"{i:0.4f}")) is None:
            result[float(f"{i:0.4f}")] = []
            color.append(cmap(int(i / max(color_value) * 256.0)))

    for i in range(len(horizontal_value)):
        result[float(f"{rounded_color_value[i]:0.4f}")].append(horizontal_value[i])

    bins: int = max(horizontal_value) - min(horizontal_value) + 1
    fig: Figure = plt.figure()

    color_bar_ax: Axes = fig.add_subplot()
    color_bar = matplotlib.colorbar.Colorbar(
        ax=color_bar_ax,
        mappable=matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(
                vmin=min(color_value), vmax=max(color_value)
            ),
            cmap=cmap,
        ),
        orientation="vertical",
    )
    color_bar.set_label(color_label, fontsize=15)
    color_bar.ax.set_position((0.95, 0.08, 0.02, 0.8))

    hist_ax: Axes = fig.add_subplot()
    hist_ax.tick_params(direction="in", which="both")
    hist_ax.hist(
        x=result.values(),  # type: ignore
        bins=bins,
        stacked=True,
        color=color,
        range=(min(horizontal_value) - 0.5, max(horizontal_value) + 0.5),
    )
    hist_ax.set_xlabel(x_label, fontsize=15)
    hist_ax.set_ylabel(y_label, fontsize=15)

    fig.savefig(f"{file}", bbox_inches="tight", transparent=transparent)
    plt.cla()
    plt.close()
