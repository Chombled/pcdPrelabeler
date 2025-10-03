"""
(internal) Functions which are used for multiple detection methods
"""

import numpy as np
from config import MIN_CLEARANCE, DBSCAN_EPSILON, DBSCAN_MIN_SAMPLES
from sklearn.cluster import DBSCAN


def get_lower_bounds(pointcloud_array):
    vehicle_mask = (
        pointcloud_array[:, 2] > 0
    )  # used since the pointcloud ships with bounding box drawn on floor
    vehicle_pointcloud = pointcloud_array[vehicle_mask]
    y_steps = np.unique(vehicle_pointcloud[:, 1])
    min_z = np.empty_like(y_steps, dtype=float)

    for i, y in enumerate(y_steps):
        step_mask = vehicle_pointcloud[:, 1] == y
        min_z[i] = vehicle_pointcloud[step_mask, 2].min()

    lower_bounds = np.column_stack((y_steps, min_z))
    return lower_bounds


def get_lowest_points(lower_bounds):
    global_min_z = lower_bounds[:, 1].min()
    wheel_mask = lower_bounds[:, 1] <= (global_min_z + MIN_CLEARANCE)
    lower_wheel_points = lower_bounds[wheel_mask]
    return lower_wheel_points


def get_dbscan_clusters(points):
    dbscan = DBSCAN(eps=DBSCAN_EPSILON, min_samples=DBSCAN_MIN_SAMPLES)
    labels = dbscan.fit_predict(points)
    return labels
