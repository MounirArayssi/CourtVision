
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--homography", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tracking = pd.read_csv(args.tracking)

    with open(args.homography) as f:
        calibration = json.load(f)

    H = np.array(calibration["homography"], dtype=np.float64)

    # Use players' feet as their estimated court contact points.
    feet = tracking[["foot_x", "foot_y"]].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1, 2)

    mapped = cv2.perspectiveTransform(feet, H).reshape(-1, 2)

    tracking["court_x_ft"] = mapped[:, 0]
    tracking["court_y_ft"] = mapped[:, 1]

    # NBA playing surface: 94 x 50 feet.
    tracking["on_court"] = (
        tracking["court_x_ft"].between(0, 94)
        & tracking["court_y_ft"].between(0, 50)
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tracking.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Total detections: {len(tracking)}")
    print(f"Mapped inside court: {tracking['on_court'].sum()}")


if __name__ == "__main__":
    main()
