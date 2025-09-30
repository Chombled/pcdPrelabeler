import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import helpers
import detection


def get_bounding_boxes(points, labels):
    unique_labels = np.unique(labels)
    clusters = [points[labels == l] for l in unique_labels]

    boxes = {}
    for label, cluster in zip(unique_labels, clusters):
        max_y = cluster[:, 1].max()
        min_x = cluster[:, 0].min()
        max_x = cluster[:, 0].max()

        anchor = (min_x, max_y)
        edge_length = max_x - min_x
        boxes[label] = (anchor, edge_length)

    return boxes


if __name__ == "__main__":
    pointcloud_array = helpers.utils.read_pointcloud()
    lower_bounds = detection.utils.get_lower_bounds(pointcloud_array)
    lower_wheel_points = detection.utils.get_lowest_points(pointcloud_array)
    labels = detection.utils.get_dbscan_clusters(lower_wheel_points)
    boxes = get_bounding_boxes(lower_wheel_points, labels)
    print(boxes)

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
    for lbl, ((anchor_x, anchor_y), edge_length) in boxes.items():
        rect = mpatches.Rectangle(
            (anchor_x, anchor_y),
            width=edge_length,
            height=edge_length,
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
