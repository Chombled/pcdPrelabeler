"""
Global variables and defaults
"""

# helper util defaults
POINTCLOUD_PATH = "/Users/emil/Documents/HSE_VSV/Project_Axle-Detection/Pointclouds/lidar_point_cloud_4"

# detection util defaults
MIN_CLEARANCE = 0.06  # maximum distance up from the lowest point to still be added to the lowest points
DBSCAN_EPSILON = 0.20  # should be >= distance between scan lines
DBSCAN_MIN_SAMPLES = 2  # should be >= 2 so the bbox isn't dimensionless

# dips defaults
INTERPOLATION_RESOLUTION = 0.01
STEP_SIZE = 0.05  # step size for the slope check
SLOPE_THRESHOLD = 0.5  # minimum slope before slope-change trigger is armed
MAX_STEPS = 5000  # safeguard against runaways