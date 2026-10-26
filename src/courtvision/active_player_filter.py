
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_track(group):
    group = group.sort_values("frame")

    frames = group["frame"].to_numpy()
    x = group["court_x_ft"].to_numpy(dtype=float)
    y = group["court_y_ft"].to_numpy(dtype=float)

    # Measure movement between consecutive observations.
    frame_gap = np.diff(frames)
    distances = np.hypot(np.diff(x), np.diff(y))

    valid = (
        (frame_gap > 0)
        & (frame_gap <= 3)
        & np.isfinite(distances)
    )

    # Approximate movement per frame in projected court feet.
    movement = (
        float(np.median(distances[valid] / frame_gap[valid]))
        if valid.any() else 0.0
    )

    on_court_rate = float(group["on_court"].mean())

    # Use frame-level evidence, not the smoothed track label.
    votes = group["frame_team"].value_counts()
    warriors = int(votes.get("warriors", 0))
    celtics = int(votes.get("celtics", 0))

    total = len(group)
    team_rate = (warriors + celtics) / total

    if warriors > celtics:
        team = "warriors"
    elif celtics > warriors:
        team = "celtics"
    else:
        team = "unknown"

    return pd.Series({
        "observations": total,
        "on_court_rate": on_court_rate,
        "movement_per_frame": movement,
        "team_evidence": team_rate,
        "candidate_team": team,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/nba/reference_classified.csv"
    )
    parser.add_argument(
        "--output",
        default="data/processed/nba/active_players.csv"
    )
    parser.add_argument(
        "--summary",
        default="data/processed/nba/active_player_summary.csv"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = [
        "frame", "track_id", "court_x_ft",
        "court_y_ft", "on_court", "frame_team"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["on_court"] = (
        df["on_court"].astype(str).str.lower() == "true"
    )

    summary = (
        df.groupby("track_id")
        .apply(analyze_track, include_groups=False)
        .reset_index()
    )

    # Experimental candidate thresholds.
    # These are intentionally adjustable, not learned rules.
    summary["active_player_candidate"] = (
        (summary["observations"] >= 20)
        & (summary["on_court_rate"] >= 0.60)
        & (
            (summary["team_evidence"] >= 0.25)
            | (summary["movement_per_frame"] >= 0.12)
        )
    )

    # Preserve all observations and original tracking IDs.
    df = df.merge(
        summary[
            ["track_id", "active_player_candidate",
             "candidate_team"]
        ],
        on="track_id",
        how="left",
        validate="many_to_one"
    )

    output = Path(args.output)
    summary_path = Path(args.summary)

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output, index=False)

    summary = summary.sort_values(
        ["active_player_candidate", "on_court_rate"],
        ascending=False
    )
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {output}")
    print(f"Summary: {summary_path}")

    print("\nTrack analysis:")
    print(summary.to_string(index=False))

    print("\nCandidates per frame:")
    counts = (
        df[df["active_player_candidate"]]
        .groupby("frame")["track_id"]
        .nunique()
    )
    print(counts.describe().to_string())


if __name__ == "__main__":
    main()
