
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# NBA court: 94 x 50 feet
# Right basket, baseline at x=94.
# Lane is 16 feet wide; free-throw line is 19 feet
# from the baseline.

LANDMARKS = [
    ("Baseline - upper lane corner", (94, 17)),
    ("Baseline - lower lane corner", (94, 33)),
    ("Free throw - lower lane corner", (75, 33)),
    ("Free throw - upper lane corner", (75, 17)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="data/processed/nba/homography.json"
    )
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Could not read video frame")

    h, w = frame.shape[:2]
    scale = min(1280 / w, 800 / h, 1.0)
    display = cv2.resize(frame, None, fx=scale, fy=scale)

    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x / scale, y / scale])

    cv2.namedWindow("NBA Court Landmarks")
    cv2.setMouseCallback("NBA Court Landmarks", on_click)

    while True:
        canvas = display.copy()

        for i, (x, y) in enumerate(points):
            px, py = round(x * scale), round(y * scale)
            cv2.circle(canvas, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(
                canvas, str(i + 1), (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )

        if len(points) < 4:
            instruction = f"Click {len(points)+1}: {LANDMARKS[len(points)][0]}"
        else:
            instruction = "Press ENTER to save"

        cv2.putText(
            canvas, instruction, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2
        )

        cv2.imshow("NBA Court Landmarks", canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (13, 10) and len(points) == 4:
            break
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    source = np.array(points, dtype=np.float32)
    destination = np.array(
        [coords for _, coords in LANDMARKS],
        dtype=np.float32
    )

    # Maps image pixels into NBA court coordinates (feet).
    H = cv2.getPerspectiveTransform(source, destination)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump({
            "video": Path(args.input).name,
            "frame": args.frame,
            "source_points": source.tolist(),
            "destination_points": destination.tolist(),
            "homography": H.tolist(),
            "court_length_ft": 94,
            "court_width_ft": 50
        }, f, indent=2)

    print(f"Saved calibration: {output}")


if __name__ == "__main__":
    main()
