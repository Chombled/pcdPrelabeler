import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import helpers
import detection


if __name__ == "__main__":
    pointcloud_array = helpers.utils.read_pointcloud()
    lower_bounds = detection.utils.get_lower_bounds(pointcloud_array)
    lower_wheel_points = detection.utils.get_lowest_points(lower_bounds)
    labels = detection.utils.get_dbscan_clusters(lower_wheel_points)
    boxes = detection.dips.get_bounding_boxes(pointcloud_array)

    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.axis("equal")

    plt.scatter(
        pointcloud_array[:, 1],
        pointcloud_array[:, 2],
        c=pointcloud_array[:, 0],
        cmap="winter",
        edgecolor="k",
        s=15,
        alpha=0.3,
        zorder=0,
    )  # plot for pointcloud with depth map
    plt.plot(
        lower_bounds[:, 0],
        lower_bounds[:, 1],
        color="cyan",
        label="Lowest height per scanline",
        zorder=5,
    )  # plot for lower vehicle bound
    plt.scatter(
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
        ax.add_patch(rect)

    plt.xlabel("y")
    plt.ylabel("z")
    plt.title("Lower vehicle bound")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.show()
