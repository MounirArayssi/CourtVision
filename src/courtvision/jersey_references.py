
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from team_classification import extract_jersey


def jersey_histogram(crop):
    """Extract a normalized HSV color histogram."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [18, 8, 8],
        [0, 180, 0, 256, 0, 256],
    )

    return cv2.normalize(
        hist, hist, norm_type=cv2.NORM_L1
    ).flatten().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="data/raw/nba_sample.mp4"
    )
    parser.add_argument(
        "--tracking",
        default="data/processed/nba/court_tracking.csv"
    )
    parser.add_argument(
        "--output",
        default="data/processed/nba/jersey_references.npz"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.tracking)

    # Manually verified examples from frame 0.
    reference_ids = {
        "warriors": [1, 2],
        "celtics": [4, 5],
        "referee": [3],
    }

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Unable to read frame 0")

    references = {}

    for team, ids in reference_ids.items():
        histograms = []

        for track_id in ids:
            rows = df[
                (df["frame"] == 0) &
                (df["track_id"] == track_id)
            ]

            if rows.empty:
                continue

            crop = extract_jersey(frame, rows.iloc[0])

            if crop is None:
                continue

            histograms.append(jersey_histogram(crop))

        if not histograms:
            raise RuntimeError(
                f"No valid references for {team}"
            )

        references[team] = np.stack(histograms)
        print(f"{team}: {len(histograms)} reference crops")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **references)

    print(f"Saved reference profiles: {output}")


if __name__ == "__main__":
    main()
