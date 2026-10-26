
import argparse
from pathlib import Path

import cv2
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--frame", type=int, default=0)
args = parser.parse_args()

video = "data/raw/nba_sample.mp4"
csv_path = "data/processed/nba/court_tracking.csv"
output = Path(f"data/processed/nba/filter_frame_{args.frame}.jpg")

df = pd.read_csv(csv_path)
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
ok, image = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read video frame")

selected = df[df["frame"] == args.frame]

for _, person in selected.iterrows():
    on_court = str(person["on_court"]).lower() == "true"

    # Green: inside mapped court; red: outside.
    color = (0, 220, 0) if on_court else (0, 0, 255)

    x = int(person["foot_x"])
    y = int(person["foot_y"])

    cv2.circle(image, (x, y), 7, color, -1)
    cv2.putText(
        image,
        str(int(person["track_id"])),
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )

output.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output), image)

print(f"Saved: {output}")
print(f"Mapped inside: {selected['on_court'].sum()}")
print(f"Total tracked: {len(selected)}")
