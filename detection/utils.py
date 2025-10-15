"""
(internal) Functions which are used for multiple detection methods
"""

import numpy as np
from config import MIN_CLEARANCE, DBSCAN_EPSILON, DBSCAN_MIN_SAMPLES
from sklearn.cluster import DBSCAN


def get_lower_bounds(pointcloud_array):
    center_z = (pointcloud_array[:, 2].min() + pointcloud_array[:, 2].max()) / 2
    vehicle_mask = (pointcloud_array[:, 1] > 0) & (pointcloud_array[:, 2] > center_z)
    vehicle_pointcloud = pointcloud_array[vehicle_mask]
    x_steps = np.unique(vehicle_pointcloud[:, 0])
    min_y = np.empty_like(x_steps, dtype=float)

    for i, x in enumerate(x_steps):
        step_mask = vehicle_pointcloud[:, 0] == x
        min_y[i] = vehicle_pointcloud[step_mask, 1].min()

    lower_bounds = np.column_stack((x_steps, min_y))
    return lower_bounds


def get_lowest_points(lower_bounds):
    global_min_y = lower_bounds[:, 1].min()
    wheel_mask = lower_bounds[:, 1] <= (global_min_y + MIN_CLEARANCE)
    lower_wheel_points = lower_bounds[wheel_mask]
    return lower_wheel_points


def get_dbscan_clusters(points):
    dbscan = DBSCAN(eps=DBSCAN_EPSILON, min_samples=DBSCAN_MIN_SAMPLES)
    labels = dbscan.fit_predict(points)
    return labels
