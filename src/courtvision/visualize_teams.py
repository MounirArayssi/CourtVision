
import argparse
from pathlib import Path

import cv2
import pandas as pd


COLORS = {
    "celtics": (0, 220, 0),       # Green
    "warriors": (255, 255, 255),  # White
    "unknown": (150, 150, 150),   # Gray
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="data/raw/nba_sample.mp4"
    )
    parser.add_argument(
        "--tracking",
        default="data/processed/nba/active_players.csv"
    )
    parser.add_argument(
        "--output",
        default="data/processed/nba/active_players_annotated.mp4"
    )
    parser.add_argument(
        "--on-court-only",
        action="store_true"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all detections, including non-candidates"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.tracking)

    required = [
        "frame", "track_id", "frame_team",
        "active_player_candidate"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    # Handle CSV boolean values consistently.
    df["active_player_candidate"] = (
        df["active_player_candidate"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    df["on_court"] = (
        df["on_court"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    grouped = {
        int(frame): group
        for frame, group in df.groupby("frame")
    }

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video")

    frame_number = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        people = grouped.get(frame_number)

        if people is not None:
            for _, person in people.iterrows():

                # Show only active-player candidates by default.
                if (
                    not args.show_all
                    and not person["active_player_candidate"]
                ):
                    continue

                if (
                    args.on_court_only
                    and not person["on_court"]
                ):
                    continue

                team = str(person["team"]).lower()

                if team not in ("celtics", "warriors"):
                    team = str(person["frame_team"]).lower()


                color = COLORS.get(team, COLORS["unknown"])

                x1, y1, x2, y2 = [
                    int(person[k])
                    for k in ("x1", "y1", "x2", "y2")
                ]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                label = (
                    f"ID {int(person['track_id'])} | {team}"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        writer.write(frame)
        frame_number += 1

    cap.release()
    writer.release()

    print(f"Saved annotated video: {output}")
    print(f"Frames processed: {frame_number}")


if __name__ == "__main__":
    main()
