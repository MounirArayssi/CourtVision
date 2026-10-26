
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def calibrate(frame):
    h, w = frame.shape[:2]
    scale = min(1200 / w, 800 / h, 1.0)
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x / scale, y / scale])

    cv2.namedWindow("Court Mapping")
    cv2.setMouseCallback("Court Mapping", on_click)

    print("Click four corners in this order:")
    print("1. Far-left")
    print("2. Far-right")
    print("3. Near-right")
    print("4. Near-left")
    print("Press R to reset, Enter to confirm, Q to quit.")

    while True:
        canvas = display.copy()

        for i, (x, y) in enumerate(points):
            p = (round(x * scale), round(y * scale))
            cv2.circle(canvas, p, 6, (0, 255, 0), -1)
            cv2.putText(
                canvas, str(i + 1), (p[0] + 8, p[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        cv2.imshow("Court Mapping", canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("r"):
            points.clear()
        elif key in (13, 10) and len(points) == 4:
            break
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            raise SystemExit("Calibration cancelled")

    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/raw/sample.mp4")
    parser.add_argument("--csv", default="data/processed/tracking.csv")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Could not read video")
    
    source_points = calibrate(frame)

    COURT_LENGTH = 28.0
    COURT_WIDTH = 15.0

    destination_points = np.array([
        [0, 0],                         # Far-left
        [COURT_LENGTH, 0],              # Far-right
        [COURT_LENGTH, COURT_WIDTH],    # Near-right
        [0, COURT_WIDTH]                # Near-left
        ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
        source_points, destination_points
    )

    df = pd.read_csv(args.csv)

    foot_positions = df[["foot_x", "foot_y"]].to_numpy(
        dtype=np.float32
    ).reshape(-1, 1, 2)

    mapped = cv2.perspectiveTransform(foot_positions, H)
    mapped = mapped.reshape(-1, 2)

    df["court_x"] = mapped[:, 0]
    df["court_y"] = mapped[:, 1]

    df["within_mapped_court"] = (
        df["court_x"].between(0, COURT_LENGTH) &
        df["court_y"].between(0, COURT_WIDTH)
    )

    df.to_csv(output / "court_tracking.csv", index=False)

    with open(output / "homography.json", "w") as f:
        json.dump({
            "source_points": source_points.tolist(),
            "destination_points": destination_points.tolist(),
            "homography": H.tolist(),
            "coordinate_system": "normalized_0_100"
        }, f, indent=2)

    print("Saved court_tracking.csv and homography.json")


if __name__ == "__main__":
    main()
