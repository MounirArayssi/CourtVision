
from pathlib import Path

import cv2
import pandas as pd

from team_classification import extract_jersey

VIDEO = "data/raw/nba_sample.mp4"
CSV = "data/processed/nba/classified_tracking.csv"
OUTPUT = Path("data/processed/nba/jersey_debug")

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
frame_data = df[df["frame"] == 0]

cap = cv2.VideoCapture(VIDEO)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read frame 0")

for _, person in frame_data.iterrows():
    crop = extract_jersey(frame, person)

    if crop is None:
        continue

    track_id = int(person["track_id"])
    label = person["frame_team"]

    cv2.imwrite(
        str(OUTPUT / f"id_{track_id}_{label}.jpg"),
        crop
    )

print(f"Saved {len(list(OUTPUT.glob('*.jpg')))} jersey crops")
