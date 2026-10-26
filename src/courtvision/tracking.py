import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO


CSV_FIELDS = [
    "frame",
    "timestamp_s",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "foot_x",
    "foot_y",
]


def process_video(
    input_path: str,
    output_dir: str,
    model_name: str = "yolo26n.pt",
    confidence_threshold: float = 0.25,
) -> None:
    """
    Detect and track people in a basketball video.

    Outputs:
        tracked.mp4  - annotated video containing bounding boxes / track IDs
        tracking.csv - frame-level tracking data
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_output = output_dir / "tracked.mp4"
    csv_output = output_dir / "tracking.csv"

    # Load YOLO detector.
    model = YOLO(model_name)

    # Open source video.
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError("Could not determine video FPS.")

    print(f"Input: {input_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {total_frames}")

    # MP4 output writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(video_output),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {video_output}")

    frame_number = 0

    try:
        with open(csv_output, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_FIELDS)

            while True:
                success, frame = cap.read()

                if not success:
                    break

                # COCO class 0 = person.
                #
                # persist=True tells the tracker that frames belong
                # to the same continuous video sequence.
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.10,
                    imgsz=1280,
                    verbose=False,
                )

                result = results[0]

                # Draw YOLO + tracking annotations.
                annotated_frame = result.plot()

                boxes = result.boxes

                if boxes is not None and boxes.id is not None:
                    coordinates = boxes.xyxy.cpu().numpy()
                    identities = boxes.id.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()

                    for box, track_id, confidence in zip(
                        coordinates,
                        identities,
                        confidences,
                    ):
                        x1, y1, x2, y2 = map(float, box)

                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        # Approximation of where the person contacts
                        # the playing surface.
                        foot_x = center_x
                        foot_y = y2

                        csv_writer.writerow(
                            [
                                frame_number,
                                frame_number / fps,
                                track_id,
                                float(confidence),
                                x1,
                                y1,
                                x2,
                                y2,
                                center_x,
                                center_y,
                                foot_x,
                                foot_y,
                            ]
                        )

                writer.write(annotated_frame)

                frame_number += 1

                if frame_number % 100 == 0:
                    print(
                        f"Processed {frame_number}/{total_frames} frames"
                    )

    finally:
        cap.release()
        writer.release()

    print()
    print(f"Finished processing {frame_number} frames.")
    print(f"Annotated video: {video_output}")
    print(f"Tracking data:   {csv_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CourtVision basketball player tracking pipeline"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input basketball video",
    )

    parser.add_argument(
        "--output",
        default="data/processed",
        help="Directory for generated files",
    )

    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Ultralytics YOLO model",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )

    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_dir=args.output,
        model_name=args.model,
        confidence_threshold=args.conf,
    )


if __name__ == "__main__":
    main()