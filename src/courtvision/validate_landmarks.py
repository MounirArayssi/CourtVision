
import cv2
import json
import numpy as np

VIDEO = "data/raw/nba_sample.mp4"
CALIBRATION = "data/processed/nba/homography.json"
OUTPUT = "data/processed/nba/landmark_validation.jpg"

with open(CALIBRATION) as f:
    calibration = json.load(f)

H = np.array(calibration["homography"], dtype=np.float64)
H_inv = np.linalg.inv(H)

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, calibration["frame"])
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Unable to read video frame")

# Four corners of the NBA free-throw lane in feet
court_points = np.array([
    [94, 17],
    [94, 33],
    [75, 33],
    [75, 17]
], dtype=np.float32).reshape(-1, 1, 2)

# Transform court coordinates back into image coordinates
image_points = cv2.perspectiveTransform(
    court_points, H_inv
).reshape(-1, 2)

points = np.round(image_points).astype(np.int32)

cv2.polylines(
    frame,
    [points],
    isClosed=True,
    color=(0, 255, 0),
    thickness=3
)

for i, (x, y) in enumerate(points):
    cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
    cv2.putText(
        frame, str(i + 1), (x + 12, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2
    )

cv2.imwrite(OUTPUT, frame)
print(f"Saved validation image: {OUTPUT}")
