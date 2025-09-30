"""
General helper functions
"""

from pypcd4 import PointCloud
from config import POINTCLOUD_PATH


def read_pointcloud(path=POINTCLOUD_PATH):
    pc = PointCloud.from_path(POINTCLOUD_PATH)
    pointcloud_array = pc.numpy(("x", "y", "z")) # the intensity is being omitted due to the sensor data missing it
    return pointcloud_array
