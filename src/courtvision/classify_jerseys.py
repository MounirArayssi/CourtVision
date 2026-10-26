
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from jersey_references import jersey_histogram
from team_classification import extract_jersey


def classify(crop, references, margin=0.04):
    sample = jersey_histogram(crop)

    scores = {}

    for team, histograms in references.items():
        similarities = [
            cv2.compareHist(
                sample,
                hist,
                cv2.HISTCMP_BHATTACHARYYA
            )
            for hist in histograms
        ]

        # Smaller distance = more similar.
        scores[team] = min(similarities)

    ordered = sorted(scores.items(), key=lambda x: x[1])
    best_team, best_distance = ordered[0]
    second_distance = ordered[1][1]

    # Ambiguous matches stay unknown.
    if second_distance - best_distance < margin:
        return "unknown", best_distance

    if best_team == "referee":
        return "unknown", best_distance

    # Conservative absolute distance threshold.
    if best_distance > 0.65:
        return "unknown", best_distance

    return best_team, best_distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", default="data/raw/nba_sample.mp4"
    )
    parser.add_argument(
        "--tracking",
        default="data/processed/nba/court_tracking.csv"
    )
    parser.add_argument(
        "--references",
        default="data/processed/nba/jersey_references.npz"
    )
    parser.add_argument(
        "--output",
        default="data/processed/nba/reference_classified.csv"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.tracking)

    with np.load(args.references) as data:
        references = {
            key: data[key] for key in data.files
        }

    df["frame_team"] = "unknown"
    df["jersey_distance"] = np.nan

    cap = cv2.VideoCapture(args.video)

    for frame_number, group in df.groupby("frame", sort=True):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = cap.read()

        if not ok:
            continue

        for index, person in group.iterrows():
            crop = extract_jersey(frame, person)

            if crop is None:
                continue

            label, distance = classify(crop, references)

            df.at[index, "frame_team"] = label
            df.at[index, "jersey_distance"] = distance

    cap.release()

    # Smooth only when there is strong evidence.
    stable_labels = {}

    for track_id, group in df.groupby("track_id"):
        votes = group["frame_team"]
        valid = votes[votes != "unknown"]

        if len(valid) < 5:
            stable_labels[track_id] = "unknown"
            continue

        counts = valid.value_counts()
        winner = counts.index[0]

        agreement = counts.iloc[0] / len(valid)
        coverage = len(valid) / len(group)

        if agreement >= 0.80 and coverage >= 0.30:
            stable_labels[track_id] = winner
        else:
            stable_labels[track_id] = "unknown"

    df["team"] = df["track_id"].map(stable_labels)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\nSaved: {output}")
    print("\nTrack-smoothed classifications:")
    print(df["team"].value_counts().to_string())

    print("\nUnique tracking IDs:")
    print(df.groupby("team")["track_id"].nunique().to_string())

    print("\nFrame 0:")
    print(
        df[df["frame"] == 0][
            ["track_id", "frame_team", "team",
             "jersey_distance", "on_court"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
