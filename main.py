import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib

import helpers
import detection
from config import POINTCLOUD_PATH


"""
Gets all pcds from a dir (no single files supported at this time)
"""


def collect_filepaths(folder: str, pattern: str = "*.pcd"):
    folder_path = pathlib.Path(folder).resolve()

    if not folder_path.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    return [str(file_path) for file_path in sorted(folder_path.rglob(pattern))]


""" 
Plots each pcd from its side as a scatter plot with depth colouring, 
overlays the lower bounds and points found and then draws the bounding boxes from the interpolated dips.
Usage: n = next pcd, p = previous pcd, q = quit
"""


def interactive_plot(filepaths):
    if not filepaths:
        raise ValueError("No file paths provided")

    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.axis("equal")

    current_index = 0

    def _plot_current():
        nonlocal current_index
        ax.clear()

        pointcloud_array = helpers.utils.read_pointcloud(filepaths[current_index])

        lower_bounds = detection.utils.get_lower_bounds(pointcloud_array)
        lower_wheel_points = detection.utils.get_lowest_points(lower_bounds)
        labels = detection.utils.get_dbscan_clusters(lower_wheel_points)
        boxes = detection.dips.get_bounding_boxes(pointcloud_array)

        ax.scatter(
            pointcloud_array[:, 0],
            pointcloud_array[:, 1],
            c=pointcloud_array[:, 2],
            cmap="winter",
            edgecolor="k",
            s=15,
            alpha=0.3,
            zorder=0,
        )  # plot for pointcloud with depth map
        ax.plot(
            lower_bounds[:, 0],
            lower_bounds[:, 1],
            color="cyan",
            label="Lowest height per scanline",
            zorder=5,
        )  # plot for lower vehicle bound
        ax.scatter(
            lower_wheel_points[:, 0],
            lower_wheel_points[:, 1],
            marker="X",
            c=labels,
            cmap="autumn",
            label="Wheel points",
            zorder=10,
        )  # plot for the detected wheel points (with cluster colouring)
        for lbl, ((anchor_x, anchor_y), width, height) in boxes.items():
            rect = mpatches.Rectangle(
                (anchor_x, anchor_y),
                width=width,
                height=height,
                fill=False,
                zorder=15,
            )
            ax.add_patch(rect)  # plot for bboxes

        ax.grid(True, alpha=0.4)

        fig.canvas.draw_idle()

    def _on_key(event):
        nonlocal current_index
        if event.key == "n":
            current_index = (current_index + 1) % len(filepaths)
            _plot_current()
        elif event.key == "p":
            current_index = (current_index - 1) % len(filepaths)
            _plot_current()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    _plot_current()
    plt.show()


def main():
    paths = collect_filepaths(POINTCLOUD_PATH)
    interactive_plot(paths)


if __name__ == "__main__":
    main()
