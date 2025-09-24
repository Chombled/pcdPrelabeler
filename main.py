import numpy as np
import matplotlib.pyplot as plt
from pypcd4 import PointCloud

POINTCLOUD_PATH = "/Users/emil/Documents/HSE_VSV/Project_Axle-Detection/Pointclouds/lidar_point_cloud_3/transit_20250922-100144_c500_id10561.pcd"
MIN_CLEARENCE = 0.05


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
    return lower_bounds[wheel_mask]


if __name__ == "__main__":
    pointcloud_array = read_pointcloud()
    lower_bounds = get_lower_bounds(pointcloud_array)
    lower_wheel_points = get_lower_wheel_points(pointcloud_array)

    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.axis("equal")

    plt.scatter(
        pointcloud_array[:, 1],
        pointcloud_array[:, 2],
        c=pointcloud_array[:, 0],
        cmap="plasma",
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
        c="red",
        label="Wheel points",
        zorder=10,
    )  # plot for the detected wheel points

    plt.xlabel("y")
    plt.ylabel("z")
    plt.title("Lower vehicle bound")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.show()
