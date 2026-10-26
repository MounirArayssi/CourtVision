
from pathlib import Path

import cv2
import pandas as pd

VIDEO = "data/raw/nba_sample.mp4"
TRACKING = "data/processed/nba/tracking.csv"
OUTPUT = Path("data/processed/nba/occlusion_debug")

FRAMES = [0, 30, 60, 90, 120]

OUTPUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(TRACKING)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open video")

for frame_number in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()

    if not ok:
        continue

    people = df[df["frame"] == frame_number]

    for _, person in people.iterrows():
        x1, y1, x2, y2 = [
            int(person[c])
            for c in ("x1", "y1", "x2", "y2")
        ]

        track_id = int(person["track_id"])

        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 255, 255), 2
        )

        cv2.putText(
            frame, f"ID {track_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 2
        )

    path = OUTPUT / f"frame_{frame_number}.jpg"
    cv2.imwrite(str(path), frame)

    print(
        f"Frame {frame_number}: "
        f"{len(people)} tracked people"
    )

cap.release()
print(f"Saved screenshots to: {OUTPUT}")
