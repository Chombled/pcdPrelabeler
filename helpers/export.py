"""
JSON exports for labeling-software & model training
"""

import json
import pathlib
import uuid
import numpy as np
import open3d as o3d
import helpers
import detection
from config import PCD_EXPORT_DIR_NAME, JSON_EXPORT_DIR_NAME


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def process_pointcloud(path: str):

    pointcloud_array = helpers.utils.read_pointcloud(path)
    vehicle_mask = pointcloud_array[:, 1] != 0
    vehicle_pointcloud = pointcloud_array[vehicle_mask]

    return vehicle_pointcloud

def write_pcd_open3d(path: str, points: np.ndarray):

    pcd = o3d.geometry.PointCloud()

    # Open3D expects Nx3 float64
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    o3d.io.write_point_cloud(
        path,
        pcd,
        write_ascii=False,
        compressed=False
    )

def export_all_bounding_boxes(pcd_dir: str):

    PCD_EXPORT_DIR = pathlib.Path(PCD_EXPORT_DIR_NAME).resolve()
    PCD_EXPORT_DIR.mkdir(exist_ok=True)

    JSON_EXPORT_DIR = pathlib.Path(JSON_EXPORT_DIR_NAME).resolve()
    JSON_EXPORT_DIR.mkdir(exist_ok=True)

    folder_path = pathlib.Path(pcd_dir).resolve()
    pcd_files = sorted(folder_path.rglob("*.pcd"))

    if not pcd_files:
        raise ValueError(f"No .pcd files found in {folder_path}")

    print(f"Found {len(pcd_files)} pointcloud files. Exporting to {JSON_EXPORT_DIR} and {PCD_EXPORT_DIR}/")

    for data_id, file_path in enumerate(pcd_files):

        # export pcds
        processed_cloud = process_pointcloud(str(file_path))
        write_pcd_open3d(PCD_EXPORT_DIR / f"{file_path.stem}.pcd", processed_cloud)

        # export jsons
        pointcloud_array = helpers.utils.read_pointcloud(str(file_path))
        boxes = detection.dips.get_bounding_boxes(pointcloud_array, is_3d=True)

        objects = []

        for track_index, (_, (anchor, width, height, depth)) in enumerate(boxes.items()):
            
            center_x = anchor[0] + width  / 2
            center_y = anchor[1] + height / 2
            center_z = anchor[2] + depth  / 2

            obj = {
                "id": str(uuid.uuid4()),
                "type": "3D_BOX",

                "classId": 9,
                "className": "Axle_Basic",

                "trackId": str(uuid.uuid4()),
                "trackName": str(track_index + 1),

                "classValues": [],

                "contour": {
                    "pointN": 0,
                    "points": [],

                    "size3D": {
                        "x": float(width),
                        "y": float(height),
                        "z": float(depth),
                    },

                    "center3D": {
                        "x": float(center_x),
                        "y": float(center_y),
                        "z": float(center_z),
                    },

                    "viewIndex": 0,

                    "rotation3D": {
                        "x": 0,
                        "y": 0,
                        "z": float(np.pi),
                    },
                },

                "modelConfidence": None,
                "modelClass": "",
            }

            objects.append(obj)

        export_data = [
            {
                "version": "Xtreme1 v0.6",
                "dataId": data_id,
                "sourceName": "Ground Truth",
                "classificationValues": None,
                "objects": objects,
            }
        ]

        # filename matches PCD stem exactly
        export_path = JSON_EXPORT_DIR / f"{file_path.stem}.json"

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2, cls=NumpyEncoder)

        print(f"Exported {len(objects)} boxes -> {export_path.name}")

    print("Export complete")
