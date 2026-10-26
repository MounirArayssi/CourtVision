
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "data/raw/nba_sample.mp4"
OUTPUT = Path("data/processed/nba/detection_comparison")
FRAMES = [0, 30, 60, 90, 120]

MODEL = "yolo26s.pt"
CONF = 0.10
IMGSZ = 1280

OUTPUT.mkdir(parents=True, exist_ok=True)


def detect(model, image):
    results = model.predict(
        image,
        classes=[0],
        conf=CONF,
        imgsz=IMGSZ,
        verbose=False,
    )[0]

    if results.boxes is None:
        return np.empty((0, 4), dtype=np.float32)

    return results.boxes.xyxy.cpu().numpy()


def draw_boxes(image, boxes):
    result = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

    return result


def main():
    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first_frame = cap.read()

    if not ok:
        raise RuntimeError("Cannot read first frame")

    print("Select a rectangle containing all visible active players.")
    print("Include the edges of the playing area.")
    print("Press ENTER to confirm.")

    roi = cv2.selectROI(
        "Select playing area",
        first_frame,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi)

    if w <= 0 or h <= 0:
        raise RuntimeError("No valid crop selected")

    model = YOLO(MODEL)

    print(f"Selected crop: x={x}, y={y}, w={w}, h={h}")

    for frame_number in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()

        if not ok:
            print(f"Skipped frame {frame_number}")
            continue

        # Baseline: predict on full image.
        full_boxes = detect(model, frame)

        # Experiment: predict on playing-area crop.
        crop = frame[y:y+h, x:x+w]
        crop_boxes = detect(model, crop)

        # Convert crop-local coordinates to full-frame coordinates.
        mapped_boxes = crop_boxes.copy()
        mapped_boxes[:, [0, 2]] += x
        mapped_boxes[:, [1, 3]] += y

        # Draw results in original image coordinates.
        full_view = draw_boxes(frame, full_boxes)
        crop_view = draw_boxes(frame, mapped_boxes)

        # Show selected crop boundary.
        cv2.rectangle(
            crop_view,
            (x, y),
            (x+w, y+h),
            (255, 0, 255),
            2,
        )

        # Resize only for side-by-side visualization.
        display_width = 960
        scale = display_width / frame.shape[1]
        display_height = round(frame.shape[0] * scale)

        full_view = cv2.resize(
            full_view, (display_width, display_height)
        )
        crop_view = cv2.resize(
            crop_view, (display_width, display_height)
        )

        cv2.putText(
            full_view,
            f"FULL FRAME: {len(full_boxes)} detections",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2,
        )

        cv2.putText(
            crop_view,
            f"COURT CROP: {len(crop_boxes)} detections",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 255), 2,
        )

        comparison = np.hstack([full_view, crop_view])

        path = OUTPUT / f"comparison_{frame_number}.jpg"
        cv2.imwrite(str(path), comparison)

        print(
            f"Frame {frame_number}: "
            f"full={len(full_boxes)}, "
            f"crop={len(crop_boxes)}"
        )

    cap.release()
    print(f"\nSaved comparisons to: {OUTPUT}")


if __name__ == "__main__":
    main()
