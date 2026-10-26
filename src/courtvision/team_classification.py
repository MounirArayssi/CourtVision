
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def extract_jersey(frame, person):
    """Extract a central torso patch, avoiding most skin and shorts."""
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = [
        float(person[k]) for k in ("x1", "y1", "x2", "y2")
    ]

    bw = x2 - x1
    bh = y2 - y1

    # Central upper torso, excluding the head.
    xa = max(0, int(x1 + bw * 0.25))
    xb = min(width, int(x1 + bw * 0.75))
    ya = max(0, int(y1 + bh * 0.20))
    yb = min(height, int(y1 + bh * 0.50))

    if xb <= xa or yb <= ya:
        return None

    crop = frame[ya:yb, xa:xb]

    if crop.size == 0:
        return None

    return crop


def jersey_features(crop):
    """Measure approximate white and green jersey coverage."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Bright, low-saturation fabric.
    white = (s < 75) & (v > 145)

    # Green fabric, with sufficient saturation.
    green = (
        (h >= 35) &
        (h <= 95) &
        (s > 45) &
        (v > 35)
    )

    white_ratio = float(np.mean(white))
    green_ratio = float(np.mean(green))

    return white_ratio, green_ratio


def classify_frame(white_ratio, green_ratio):
    """
    Conservative initial heuristic.
    Thresholds are baselines, not trained probabilities.
    """
    if white_ratio >= 0.35 and white_ratio > green_ratio * 2:
        return "warriors"

    if green_ratio >= 0.12 and green_ratio > white_ratio:
        return "celtics"

    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.tracking)
    df["white_ratio"] = np.nan
    df["green_ratio"] = np.nan
    df["frame_team"] = "unknown"

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # Read video sequentially and classify every detection.
    for frame_number, group in df.groupby("frame", sort=True):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        success, frame = cap.read()

        if not success:
            print(f"Skipping unreadable frame {frame_number}")
            continue

        for index, person in group.iterrows():
            crop = extract_jersey(frame, person)

            if crop is None:
                continue

            white_ratio, green_ratio = jersey_features(crop)
            label = classify_frame(white_ratio, green_ratio)

            df.at[index, "white_ratio"] = white_ratio
            df.at[index, "green_ratio"] = green_ratio
            df.at[index, "frame_team"] = label

    cap.release()

    # Smooth team labels across observations sharing an ID.
    stable_labels = {}

    for track_id, group in df.groupby("track_id"):
        votes = group["frame_team"]
        confident = votes[votes != "unknown"]

        if len(confident) < 3:
            stable_labels[track_id] = "unknown"
            continue

        counts = confident.value_counts()
        winning_team = counts.index[0]
        agreement = counts.iloc[0] / len(confident)
        coverage = len(confident) / len(group)

        # Require strong agreement and some evidence over time.
        if agreement >= 0.80 and coverage >= 0.25:
            stable_labels[track_id] = winning_team
        else:
            stable_labels[track_id] = "unknown"

    df["team"] = df["track_id"].map(stable_labels)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nSaved: {output}")
    print("\nTeam observations:")
    print(df["team"].value_counts().to_string())

    print("\nUnique track IDs per classification:")
    print(df.groupby("team")["track_id"].nunique().to_string())

    print("\nFrame 0 classifications:")
    print(
        df[df["frame"] == 0][
            ["track_id", "team", "frame_team",
             "white_ratio", "green_ratio", "on_court"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
