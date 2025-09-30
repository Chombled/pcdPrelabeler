"""
Draws bounding boxes from the clustered lowest points.
The distance between the edge points along the x-axis (lenght of the vehicle) is used as the height and width of the bbox.
The anchor of the bbox is the highest point of the two edge points. (bottom left or right point)
"""

import numpy as np


def get_bounding_boxes(points, labels):
    unique_labels = np.unique(labels)
    clusters = [points[labels == label] for label in unique_labels]

    boxes = {}
    for label, cluster in zip(unique_labels, clusters):
        max_y = cluster[:, 1].max()
        min_x = cluster[:, 0].min()
        max_x = cluster[:, 0].max()

        anchor = (min_x, max_y)
        edge_length = max_x - min_x
        boxes[label] = (anchor, edge_length)

    return boxes
