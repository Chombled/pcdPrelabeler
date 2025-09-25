import numpy as np
import matplotlib.pyplot as plt
from pypcd4 import PointCloud
from sklearn.cluster import DBSCAN
import matplotlib.patches as mpatches

POINTCLOUD_PATH = "/Users/emil/Documents/HSE_VSV/Project_Axle-Detection/Pointclouds/lidar_point_cloud_3/transit_20250922-100144_c500_id10561.pcd"
MIN_CLEARENCE = 0.06  # 11.5 cm by "law" but mudflaps exist :(
DBSCAN_EPSILON = 0.15  # scanlines are about 12 cm apart
DBSCAN_MIN_SAMPLES = 2  # needs two points for a bounding box


def read_pointcloud(path=POINTCLOUD_PATH):
    pc = PointCloud.from_path(POINTCLOUD_PATH)
    pointcloud_array = pc.numpy(("x", "y", "z"))
    return pointcloud_array


def get_lower_bounds(pointcloud_array):
    vehicle_mask = pointcloud_array[:, 2] > 0
    vehicle_pointcloud = pointcloud_array[vehicle_mask]
    y_steps = np.unique(vehicle_pointcloud[:, 1])
    min_z = np.empty_like(y_steps, dtype=float)

    for i, y in enumerate(y_steps):
        step_mask = vehicle_pointcloud[:, 1] == y
        min_z[i] = vehicle_pointcloud[step_mask, 2].min()

    lower_bounds = np.column_stack((y_steps, min_z))
    return lower_bounds


def get_lower_wheel_points(pointcloud_array):
    lower_bounds = get_lower_bounds(pointcloud_array)
    global_min_z = lower_bounds[:, 1].min()
    wheel_mask = lower_bounds[:, 1] <= (global_min_z + MIN_CLEARENCE)
    lower_wheel_points = lower_bounds[wheel_mask]
    return lower_wheel_points


def get_point_cluster(lower_wheel_points):
    dbscan = DBSCAN(eps=DBSCAN_EPSILON, min_samples=DBSCAN_MIN_SAMPLES)
    labels = dbscan.fit_predict(lower_wheel_points)
    return labels


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
    pointcloud_array = read_pointcloud()
    lower_bounds = get_lower_bounds(pointcloud_array)
    lower_wheel_points = get_lower_wheel_points(pointcloud_array)
    labels = get_point_cluster(lower_wheel_points)
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
