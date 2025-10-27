"""
JSON exports for labeling-software & model training
"""

import json
import os
import pathlib
import numpy as np
import helpers
import detection

EXPORT_DIR = pathlib.Path("exports").resolve()
EXPORT_DIR.mkdir(exist_ok=True)


# since the JSON can't parse numpy Dtypes :(
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def export_all_bounding_boxes(pcd_dir: str):
    folder_path = pathlib.Path(pcd_dir).resolve()
    pcd_files = sorted(folder_path.rglob("*.pcd"))  # all .pcd files in the folder
    if not pcd_files:
        raise ValueError(f"No .pcd files found in {folder_path}")

    print(f"Found {len(pcd_files)} pointcloud files. Exporting to {EXPORT_DIR}/")

    for file_path in pcd_files:
        # aquires raw and detection data
        pointcloud_array = helpers.utils.read_pointcloud(str(file_path))
        boxes = detection.dips.get_bounding_boxes(pointcloud_array, is_3d=True)

        # export format (placeholder), currently exports one file per pcd with all relevant info about the bbox positions
        export_data = {
            "name": os.path.basename(file_path),
            "boxes": [
                {
                    "id": str(lbl),  # ensure label is a string
                    "anchor": [float(a) for a in anchor],
                    "extent": [float(width), float(height), float(depth)],
                    "class": "feature",
                }
                for lbl, (anchor, width, height, depth) in boxes.items()
            ],
        }

        # exports folder is filled with the same names as the import just with an appendix
        export_path = EXPORT_DIR / (file_path.stem + "_labels.json")
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2, cls=NumpyEncoder)

        print(f"Exported {len(boxes)} bounding boxes for {file_path.name}")

    print("Export complete")
