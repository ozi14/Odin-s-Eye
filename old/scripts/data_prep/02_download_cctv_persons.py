"""
Download CCTV Persons dataset from Roboflow.

1. pip install roboflow
2. export ROBOFLOW_API_KEY="your_key_here"
3. python scripts/data_prep/02_download_cctv_persons.py
"""

import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT = os.path.join(BASE, "datasets", "cctv_persons")


if __name__ == "__main__":
    from roboflow import Roboflow

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY environment variable before running.\n"
            "  export ROBOFLOW_API_KEY='your_key_here'"
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("car-vision").project("cctv-persons")
    dataset = project.version(1).download("yolov11", location=OUTPUT)

    print(f"\n✓ CCTV Persons downloaded to: {OUTPUT}")
