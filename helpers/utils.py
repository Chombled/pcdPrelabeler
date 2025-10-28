"""
General helper functions
"""

from pypcd4 import PointCloud

"""
Reads the pcd so the x, y and z directions line up with the 2d projection (x = lenght, y = height, z = width of the vehicle)
"""


def read_pointcloud(path):
    pc = PointCloud.from_path(path)
    pointcloud_array = pc.numpy(
        ("y", "z", "x")
    )  # the intensity is being omitted due to the sensor data missing it
    return pointcloud_array
