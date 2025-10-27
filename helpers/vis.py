"""
Single pointcloud interactive visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import helpers
import detection


def interactive_plot(filepaths):

    if not filepaths:
        raise ValueError("No filepaths provided")

    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.axis("equal")

    index = 0

    def _plot_current():
        nonlocal index
        ax.clear()

        # read pcd
        file_path = filepaths[index]
        pointcloud_array = helpers.utils.read_pointcloud(file_path)

        # aquires data from detection functions
        lower_bounds = detection.utils.get_lower_bounds(pointcloud_array)
        lower_wheel_points = detection.utils.get_lowest_points(lower_bounds)
        labels = detection.utils.get_dbscan_clusters(lower_wheel_points)
        boxes = detection.dips.get_bounding_boxes(pointcloud_array)

        # scatter plot of the unedited pointcloud
        ax.scatter(
            pointcloud_array[:, 0],
            pointcloud_array[:, 1],
            c=pointcloud_array[:, 2],
            cmap="winter",
            edgecolor="k",
            s=15,
            alpha=0.3,
            zorder=0,
        )

        # plot of the lowest bounds not including the ground box
        ax.plot(
            lower_bounds[:, 0],
            lower_bounds[:, 1],
            color="cyan",
            label="Lowest height per scanline",
            zorder=5,
        )

        # scatter plot of the lowest points (lowest points + grace distance) with cluster colouring
        ax.scatter(
            lower_wheel_points[:, 0],
            lower_wheel_points[:, 1],
            marker="X",
            c=labels,
            cmap="autumn",
            label="Wheel points",
            zorder=10,
        )

        # bounding boxes over the expanded wheel clusters
        for lbl, ((anchor_x, anchor_y), width, height) in boxes.items():
            rect = mpatches.Rectangle(
                (anchor_x, anchor_y),
                width=width,
                height=height,
                fill=False,
                zorder=15,
            )
            ax.add_patch(rect)

        ax.set_title(f"{index + 1}/{len(filepaths)} — {file_path}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.4)
        fig.canvas.draw_idle()

    def _on_key(event):
        nonlocal index

        # next
        if event.key == "n":
            index = (index + 1) % len(filepaths)
            _plot_current()

        # previous
        elif event.key == "p":
            index = (index - 1) % len(filepaths)
            _plot_current()

        # quit
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)

    _plot_current()
    plt.show()
