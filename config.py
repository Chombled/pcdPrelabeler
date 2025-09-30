"""
Global variables and defaults
"""

POINTCLOUD_PATH = "/Users/emil/Documents/HSE_VSV/Project_Axle-Detection/Pointclouds/lidar_point_cloud_3/transit_20250922-100144_c500_id10561.pcd"
MIN_CLEARANCE = 0.06  # 11.5 cm by "law" but mudflaps exist :(
DBSCAN_EPSILON = 0.15  # scanlines are about 12 cm apart
DBSCAN_MIN_SAMPLES = 2  # needs two points for a bounding box
