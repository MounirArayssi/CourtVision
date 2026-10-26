
import cv2
import json
import numpy as np
from pathlib import Path

VIDEO_PATH = "data/raw/sample.mp4"
COURT_PATH = "data/processed/court.json"
OUTPUT_PATH = "data/processed/court_overlay.jpg"

with open(COURT_PATH, "r") as f:
    court = json.load(f)

polygon = np.array(
    court["court_polygon"],
    dtype=np.int32
)

cap = cv2.VideoCapture(VIDEO_PATH)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame")

# Draw a translucent green region.
overlay = frame.copy()

cv2.fillPoly(
    overlay,
    [polygon],
    color=(0, 255, 0)
)

frame = cv2.addWeighted(
    overlay, 0.2,
    frame, 0.8,
    0
)

# Draw the polygon outline.
cv2.polylines(
    frame,
    [polygon],
    isClosed=True,
    color=(0, 255, 0),
    thickness=5
)

for i, (x, y) in enumerate(polygon):
    cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)
    cv2.putText(
        frame,
        str(i + 1),
        (int(x) + 15, int(y) - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3
    )

output = Path(OUTPUT_PATH)
output.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output), frame)

print(f"Court overlay saved to {output}")
