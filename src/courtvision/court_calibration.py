
import cv2
import json
import argparse
from pathlib import Path

points = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="data/processed/court.json"
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Could not read frame")

    height, width = frame.shape[:2]

    # Display at a manageable size; save original coordinates.
    scale = min(1.0, 1280 / width, 800 / height)
    display = cv2.resize(
        frame, None, fx=scale, fy=scale
    )

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([
                round(x / scale),
                round(y / scale)
            ])

    cv2.namedWindow("Court Calibration")
    cv2.setMouseCallback("Court Calibration", on_click)

    while True:
        canvas = display.copy()

        scaled_points = [
            (round(x * scale), round(y * scale))
            for x, y in points
        ]

        for p in scaled_points:
            cv2.circle(canvas, p, 5, (0, 255, 0), -1)

        for i in range(1, len(scaled_points)):
            cv2.line(
                canvas,
                scaled_points[i - 1],
                scaled_points[i],
                (0, 255, 0),
                2
            )

        if len(scaled_points) >= 3:
            cv2.line(
                canvas,
                scaled_points[-1],
                scaled_points[0],
                (0, 255, 0),
                2
            )

        cv2.imshow("Court Calibration", canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("u") and points:
            points.pop()

        elif key == ord("s"):
            if len(points) < 3:
                print("Select at least 3 points")
                continue

            output = Path(args.output)
            output.parent.mkdir(
                parents=True, exist_ok=True
            )

            data = {
                "video": Path(args.input).name,
                "coordinate_system": "original_pixels",
                "frame_width": width,
                "frame_height": height,
                "court_polygon": points
            }

            with open(output, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Saved court boundaries to {output}")
            break

        elif key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
