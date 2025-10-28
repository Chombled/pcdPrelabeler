import argparse
import pathlib
from config import POINTCLOUD_PATH
from helpers.vis import interactive_plot
from helpers.export import export_all_bounding_boxes


def collect_filepaths(folder: str, pattern: str = "*.pcd"):
    folder_path = pathlib.Path(folder).resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")
    return [str(file_path) for file_path in sorted(folder_path.rglob(pattern))]


def main():
    parser = argparse.ArgumentParser(
        description="vehicle pointcloud prelabeler with previewer"
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["view", "export"],
        default="view",
        help="view or export the pointclouds from a folder",
    )
    args = parser.parse_args()

    if args.mode == "view":
        paths = collect_filepaths(POINTCLOUD_PATH)
        interactive_plot(paths)
    elif args.mode == "export":
        export_all_bounding_boxes(POINTCLOUD_PATH)


if __name__ == "__main__":
    main()
