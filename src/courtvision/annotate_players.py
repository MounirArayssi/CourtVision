
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROLES = {
    ord("1"): "celtics",
    ord("2"): "warriors",
    ord("3"): "referee",
    ord("4"): "bench",
    ord("5"): "spectator",
    ord("6"): "uncertain",
}

COLORS = {
    "celtics": (0, 210, 0),
    "warriors": (255, 255, 255),
    "referee": (0, 165, 255),
    "bench": (255, 100, 0),
    "spectator": (140, 140, 140),
    "uncertain": (0, 0, 255),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video", default="data/raw/nba_sample.mp4"
)
parser.add_argument(
    "--tracking",
    default="data/processed/nba/tracking.csv"
)
parser.add_argument(
    "--output",
    default="data/processed/nba/ground_truth.csv"
)
parser.add_argument("--frames", type=int, default=10)
args = parser.parse_args()

df = pd.read_csv(args.tracking)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(args.video)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_numbers = np.unique(
    np.linspace(
        0, total_frames - 1,
        args.frames,
        dtype=int
    )
)

columns = ["frame", "track_id", "role"]
if output.exists():
    labels = pd.read_csv(output).to_dict("records")
else:
    labels = []

# Use frame + track ID as the annotation key.
annotations = {
    (int(r["frame"]), int(r["track_id"])): str(r["role"])
    for r in labels
}

window = "CourtVision Annotation"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 1280, 800)

selected_id = None
current_boxes = {}
scale = 1.0


def save():
    rows = [
        {"frame": f, "track_id": tid, "role": role}
        for (f, tid), role in annotations.items()
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        output, index=False
    )


def on_mouse(event, x, y, flags, userdata):
    global selected_id

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    px, py = x / scale, y / scale

    matches = []
    for tid, (x1, y1, x2, y2) in current_boxes.items():
        if x1 <= px <= x2 and y1 <= py <= y2:
            area = (x2 - x1) * (y2 - y1)
            matches.append((area, tid))

    if matches:
        # Prefer the smallest matching box in overlaps.
        selected_id = min(matches)[1]


cv2.setMouseCallback(window, on_mouse)

position = 0

while 0 <= position < len(frame_numbers):
    frame_no = int(frame_numbers[position])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()

    if not ok:
        position += 1
        continue

    people = df[df["frame"] == frame_no]
    current_boxes = {}

    for _, person in people.iterrows():
        tid = int(person["track_id"])
        current_boxes[tid] = tuple(
            float(person[c]) for c in ("x1", "y1", "x2", "y2")
        )

    h, w = frame.shape[:2]
    scale = min(1280 / w, 760 / h, 1.0)

    while True:
        canvas = frame.copy()

        for tid, coords in current_boxes.items():
            x1, y1, x2, y2 = map(int, coords)
            role = annotations.get((frame_no, tid))

            color = COLORS.get(role, (190, 190, 190))
            thickness = 4 if tid == selected_id else 2

            cv2.rectangle(
                canvas, (x1, y1), (x2, y2),
                color, thickness
            )

            text = f"{tid}: {role or '?'}"
            cv2.putText(
                canvas, text, (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2
            )

        # Display a compact help banner.
        cv2.rectangle(
            canvas, (0, 0), (w, 95),
            (20, 20, 20), -1
        )

        instructions = [
            "Click a box, then press:",
            "1 Celtics | 2 Warriors | 3 Ref | "
            "4 Bench | 5 Fan | 6 Uncertain",
            "N Next | P Previous | D Delete label | "
            "S Save | Q Quit"
        ]

        for i, text in enumerate(instructions):
            cv2.putText(
                canvas, text, (20, 27 + i * 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )

        resized = cv2.resize(
            canvas, None, fx=scale, fy=scale
        )
        cv2.imshow(window, resized)

        key = cv2.waitKey(0) & 0xFF

        if key in ROLES and selected_id is not None:
            annotations[(frame_no, selected_id)] = ROLES[key]
            save()

        elif key == ord("d") and selected_id is not None:
            annotations.pop((frame_no, selected_id), None)
            save()

        elif key == ord("s"):
            save()
            print(f"Saved {len(annotations)} annotations")

        elif key == ord("n"):
            save()
            selected_id = None
            position += 1
            break

        elif key == ord("p"):
            save()
            selected_id = None
            position = max(0, position - 1)
            break

        elif key in (ord("q"), 27):
            save()
            cap.release()
            cv2.destroyAllWindows()
            print(f"Saved: {output}")
            raise SystemExit

save()
cap.release()
cv2.destroyAllWindows()
print(f"Completed. Saved: {output}")
