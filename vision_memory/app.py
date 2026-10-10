from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import cv2
import torch
from ultralytics import YOLO, YOLOE

from .memory import Detection, MemoryStore
from .robust import FrameAssociator, brighten_if_dark, merge_detections, tile_regions


DEFAULT_CLASSES = ["pen", "wristwatch", "set of keys", "wallet"]
OPEN_IMAGES_CLASSES = {"pen": "Pen", "wristwatch": "Watch"}
TRACKER_CONFIG = Path(__file__).resolve().parent / "pen_tracker.yaml"
COLORS = {
    "pen": (255, 170, 0),
    "wristwatch": (180, 70, 255),
    "set of keys": (0, 210, 255),
    "wallet": (70, 220, 70),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and remember everyday objects using a webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated object classes",
    )
    parser.add_argument("--model", default="yoloe-26n-seg.pt")
    parser.add_argument(
        "--open-images-model",
        default="yolov8s-oiv7.pt",
        help="Open Images model trained on Pen and Watch",
    )
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--tile-image-size",
        type=int,
        default=960,
        help="Inference size for scheduled crop scans in robust mode",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Run full-frame and tiled inference for smaller, darker objects",
    )
    parser.add_argument("--tile-grid", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--tile-interval",
        type=int,
        default=4,
        help="Run expensive tiled scans every N frames in robust mode (1 = every frame)",
    )
    parser.add_argument("--device", default="auto", help="auto, mps, cpu, or CUDA id")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--manual-pen",
        action="store_true",
        help="Skip automatic detection and track a pen selected with p",
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Do not mirror the webcam image"
    )
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_classes(value: str) -> List[str]:
    classes = [item.strip() for item in value.split(",") if item.strip()]
    if not classes:
        raise ValueError("At least one object class is required.")
    return classes


def detections_from_result(result, label_map=None) -> List[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    track_ids = (
        boxes.id.detach().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
    )

    detections: List[Detection] = []
    for xyxy, confidence, class_id, track_id in zip(
        coordinates, confidences, class_ids, track_ids
    ):
        label = str(result.names[int(class_id)])
        if label_map is not None:
            label = label_map.get(label, label)
        detections.append(
            Detection(
                label=label,
                confidence=float(confidence),
                bbox=tuple(int(round(value)) for value in xyxy),
                track_id=int(track_id) if track_id is not None else None,
            )
        )
    return detections


def predict_robust(
    model, frame, args, device, class_ids=None, label_map=None, scan_tiles=True
):
    """Detect on the original frame and overlapping crops, then map to frame pixels."""
    options = {
        "conf": args.confidence,
        "imgsz": args.image_size,
        "device": device,
        "verbose": False,
    }
    if class_ids is not None:
        options["classes"] = class_ids

    detections = detections_from_result(model.predict(frame, **options)[0], label_map)
    if not scan_tiles:
        return merge_detections(detections)
    height, width = frame.shape[:2]
    enhanced = brighten_if_dark(frame)
    tile_options = {**options, "imgsz": args.tile_image_size}
    for x1, y1, x2, y2 in tile_regions(width, height, args.tile_grid):
        tile = enhanced[y1:y2, x1:x2]
        for item in detections_from_result(
            model.predict(tile, **tile_options)[0], label_map
        ):
            box = item.bbox
            detections.append(
                replace(
                    item,
                    bbox=(box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1),
                    source="tiled model",
                )
            )
    return merge_detections(detections)


def draw_detections(frame, detections: Sequence[Detection]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = COLORS.get(detection.label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if detection.source == "manual":
            text = f"{detection.label} (selected)"
        elif detection.source == "recent":
            text = f"{detection.label} (recent, unconfirmed)"
        else:
            track = f" #{detection.track_id}" if detection.track_id is not None else ""
            text = f"{detection.label}{track} {detection.confidence:.0%}"
        cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + 10 + len(text) * 9, y1), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 5, max(16, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )


def draw_status(
    frame,
    classes: Sequence[str],
    detections: Sequence[Detection],
    recent: Sequence[Detection] = (),
) -> None:
    visible = {detection.label for detection in detections}
    recently_seen = {detection.label for detection in recent}
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (20, 20, 20), -1)
    parts = [
        f"{label}: {'VISIBLE' if label in visible else 'RECENT' if label in recently_seen else '--'}"
        for label in classes
    ]
    cv2.putText(
        frame,
        " | ".join(parts),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {index}. Check camera permissions and close "
            "other apps using the camera."
        )
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width} x {actual_height}")
    return camera


def run(args: argparse.Namespace) -> None:
    if args.tile_interval < 1:
        raise ValueError("--tile-interval must be at least 1.")
    if args.image_size < 32 or args.tile_image_size < 32:
        raise ValueError("Inference image sizes must be at least 32.")
    classes = ["pen"] if args.manual_pen else parse_classes(args.classes)
    manual_label = "pen" if "pen" in classes else classes[0]
    open_images_model = None
    open_images_ids = []
    open_images_label_map = {}
    model = None
    if args.manual_pen:
        print("Manual pen mode. Press p in the camera window to select the pen.")
    else:
        device = select_device(args.device)
        print(f"Looking for: {', '.join(classes)}")
        open_images_targets = [
            label for label in classes if label in OPEN_IMAGES_CLASSES
        ]
        prompted_targets = [
            label for label in classes if label not in OPEN_IMAGES_CLASSES
        ]
        if open_images_targets:
            print(f"Loading trained detector {args.open_images_model} on {device}...")
            open_images_model = YOLO(args.open_images_model)
            open_images_label_map = {
                OPEN_IMAGES_CLASSES[label]: label for label in open_images_targets
            }
            open_images_ids = [
                class_id
                for class_id, name in open_images_model.names.items()
                if name in open_images_label_map
            ]
            if len(open_images_ids) != len(open_images_targets):
                raise ValueError(
                    f"{args.open_images_model} does not contain all requested "
                    "Open Images classes."
                )
        if prompted_targets:
            print(f"Loading open-vocabulary detector {args.model} on {device}...")
            model = YOLOE(args.model)
            model.set_classes(prompted_targets)
    memory = MemoryStore(args.data_dir)
    camera = open_camera(args.camera, args.camera_width, args.camera_height)

    window_name = "VisionMemory - c to capture, q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    previous_time = time.monotonic()
    manual_pen_tracker = None
    robust_associator = FrameAssociator() if args.robust else None
    frame_number = 0

    try:
        while True:
            frame_number += 1
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()

            detections = []
            if open_images_model is not None:
                if args.robust:
                    detections.extend(
                        predict_robust(
                            open_images_model,
                            frame,
                            args,
                            device,
                            class_ids=open_images_ids,
                            label_map=open_images_label_map,
                            scan_tiles=(frame_number - 1) % args.tile_interval == 0,
                        )
                    )
                else:
                    result = open_images_model.track(
                        frame,
                        persist=True,
                        tracker=str(TRACKER_CONFIG),
                        classes=open_images_ids,
                        conf=args.confidence,
                        imgsz=args.image_size,
                        device=device,
                        verbose=False,
                    )[0]
                    detections.extend(
                        detections_from_result(result, open_images_label_map)
                    )
            if model is not None:
                if args.robust:
                    detections.extend(
                        predict_robust(
                            model,
                            frame,
                            args,
                            device,
                            scan_tiles=(
                                frame_number - 1 + args.tile_interval // 2
                            ) % args.tile_interval
                            == 0,
                        )
                    )
                else:
                    result = model.track(
                        frame,
                        persist=True,
                        tracker=str(TRACKER_CONFIG),
                        conf=args.confidence,
                        imgsz=args.image_size,
                        device=device,
                        verbose=False,
                    )[0]
                    detections.extend(detections_from_result(result))
            if robust_associator is not None:
                detections = robust_associator.update(
                    merge_detections(detections), frame.shape[1], frame.shape[0]
                )
            if manual_pen_tracker is not None:
                tracked, region = manual_pen_tracker.update(clean_frame)
                if tracked:
                    x, y, width, height = (int(round(value)) for value in region)
                    detections = [item for item in detections if item.label != manual_label]
                    detections.append(
                        Detection(
                            label=manual_label,
                            confidence=None,
                            bbox=(x, y, x + width, y + height),
                            track_id=None,
                            source="manual",
                        )
                    )
                else:
                    manual_pen_tracker = None
                    print("Lost the selected pen. Press p to select it again.")
            observed_at = datetime.now().astimezone()
            memory.update(detections, clean_frame, observed_at)

            recent = (
                robust_associator.recent(detections, args.tile_interval - 1)
                if robust_associator is not None
                else []
            )
            draw_detections(frame, [*recent, *detections])
            draw_status(frame, classes, detections, recent)
            now = time.monotonic()
            fps = 1 / max(now - previous_time, 0.001)
            previous_time = now
            cv2.putText(
                frame,
                f"{fps:.1f} FPS",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                print(memory.summary())
            if key == ord("c"):
                diagnostics_dir = args.data_dir / "diagnostics"
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                image_path = diagnostics_dir / (
                    f"camera_{observed_at.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                )
                if cv2.imwrite(str(image_path), clean_frame):
                    print(f"Saved camera frame: {image_path.resolve()}")
                else:
                    print(f"Could not save camera frame: {image_path.resolve()}")
            if key == ord("p"):
                region = cv2.selectROI(window_name, clean_frame, False, True)
                if region[2] > 0 and region[3] > 0:
                    manual_pen_tracker = cv2.TrackerMIL_create()
                    manual_pen_tracker.init(clean_frame, region)
                    print("Pen selected. Tracking it until it leaves view.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(memory.summary())


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped.")
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
