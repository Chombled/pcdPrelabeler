"""
General helper functions
"""

from pypcd4 import PointCloud


def read_pointcloud(path):
    pc = PointCloud.from_path(path)
    pointcloud_array = pc.numpy(
        ("x", "y", "z")
    )  # the intensity is being omitted due to the sensor data missing it
    return pointcloud_array
