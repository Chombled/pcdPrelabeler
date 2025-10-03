"""
Draws bounding boxes from the clustered lowest points together with a slope analysis for each cluster.
The lowest point clusters get expanded until a change of slope occurs after a steep slope.
The bounding boxes are anchored by the edge points of the expanded clusters in the x direction. (width = x-distance between edge points)
The bounding boxes are anchored by the lowest point of the expanded clusters in the y direction. (height = y-distance between edge points and lowest point * 2)

"""

import numpy as np
from . import utils
from scipy.interpolate import CubicSpline
from config import INTERPOLATION_RESOLUTION, STEP_SIZE, SLOPE_THRESHOLD, MAX_STEPS


def _get_spline(lower_bounds):
    spline = CubicSpline(lower_bounds[:, 0], lower_bounds[:, 1])
    return spline


def _get_interpolation_x(lower_bounds):
    x_min, x_max = lower_bounds[:, 0].min(), lower_bounds[:, 0].max()
    interpolation_x = np.arange(x_min, x_max, INTERPOLATION_RESOLUTION)
    return interpolation_x


def _walk_outwards_until_threshold(
    spline,
    start_x,
    step_size=STEP_SIZE,
    domain_min_x=None,
    domain_max_x=None,
    slope_threshold=SLOPE_THRESHOLD,
    max_steps=MAX_STEPS,
    reverse_direction=False,
):
    out_xs, out_ys = [], []

    initial_slope = spline.derivative(1)(start_x)
    previous_sign = np.sign(initial_slope)

    slope_has_reached_threshold = False
    current_x = start_x

    if domain_min_x is None:
        domain_min_x = spline.x[0]
    if domain_max_x is None:
        domain_max_x = spline.x[-1]

    for _ in range(max_steps):
        current_x += step_size

        if current_x < domain_min_x or current_x > domain_max_x:
            break

        current_slope = spline.derivative(1)(current_x)
        abs_current_slope = abs(current_slope)

        if not slope_has_reached_threshold:
            if abs_current_slope >= slope_threshold:
                slope_has_reached_threshold = True
                previous_sign = np.sign(current_slope)
            out_xs.append(current_x)
            out_ys.append(spline(current_x))
            continue

        if reverse_direction:
            if previous_sign < 0 and np.sign(current_slope) > 0:
                break
        else:
            if previous_sign > 0 and np.sign(current_slope) < 0:
                break

        previous_sign = np.sign(current_slope)
        out_xs.append(current_x)
        out_ys.append(spline(current_x))

    return np.array(out_xs), np.array(out_ys)


def _expand_clusters_with_threshold(
    cluster_list,
    spline,
    step_size=STEP_SIZE,
    slope_threshold=SLOPE_THRESHOLD,
    max_steps=MAX_STEPS,
):

    domain_min_x, domain_max_x = spline.x[0], spline.x[-1]
    expanded_clusters = []

    for cluster in cluster_list:
        sorted_cluster = cluster[cluster[:, 0].argsort()]
        cluster_min_x, cluster_max_x = (
            sorted_cluster[:, 0].min(),
            sorted_cluster[:, 0].max(),
        )

        left_x, left_y = _walk_outwards_until_threshold(
            spline,
            start_x=cluster_min_x,
            step_size=-step_size,
            domain_min_x=domain_min_x,
            domain_max_x=domain_max_x,
            slope_threshold=slope_threshold,
            max_steps=max_steps,
            reverse_direction=True,
        )

        right_x, right_y = _walk_outwards_until_threshold(
            spline,
            start_x=cluster_max_x,
            step_size=step_size,
            domain_min_x=domain_min_x,
            domain_max_x=domain_max_x,
            slope_threshold=slope_threshold,
            max_steps=max_steps,
            reverse_direction=False,
        )

        new_x = np.concatenate([left_x[::-1], sorted_cluster[:, 0], right_x])
        new_y = np.concatenate([left_y[::-1], sorted_cluster[:, 1], right_y])

        unique_mask = np.concatenate([[True], np.diff(new_x) > 1e-12])
        expanded_clusters.append(
            np.column_stack([new_x[unique_mask], new_y[unique_mask]])
        )

    return expanded_clusters


def get_bounding_boxes(pointcloud_array):
    lower_bounds = utils.get_lower_bounds(pointcloud_array)

    spline = _get_spline(lower_bounds)
    interpolation_x = _get_interpolation_x(lower_bounds)
    interpolated_lower_bounds = np.column_stack(
        (interpolation_x, spline(interpolation_x))
    )

    lowest_points = utils.get_lowest_points(interpolated_lower_bounds)
    labels = utils.get_dbscan_clusters(lowest_points)
    unique_labels = np.unique(labels)
    clusters = [lowest_points[labels == label] for label in unique_labels]

    expanded_clusters = _expand_clusters_with_threshold(clusters, spline)

    boxes = {}
    for label, cluster in zip(unique_labels, expanded_clusters):
        min_x = cluster[:, 0].min()
        max_x = cluster[:, 0].max()

        idx_min = np.argmin(cluster[:, 0])
        idx_max = np.argmax(cluster[:, 0])

        left_y = cluster[idx_min, 1]
        right_y = cluster[idx_max, 1]

        min_y = cluster[:, 1].min()

        anchor = (min_x, min_y)
        width = max_x - min_x
        height = left_y + right_y
        boxes[label] = (anchor, width, height)

    return boxes
