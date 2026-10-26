"""Photo detection with broad trained coverage and supplementary text targets."""
from dataclasses import asdict
from datetime import datetime
import json

import cv2
from ultralytics import YOLO, YOLOE

from .app import draw_detections, parse_classes, predict_robust, select_device
from .robust import box_iou

EXTRA_CLASSES = ["set of keys", "wallet", "pencil", "charging cable", "phone charger", "earbuds"]
ALIASES = {"wristwatch": "watch", "phone": "mobile phone", "smartphone": "mobile phone"}


def suppress_duplicates(detections, threshold=0.45):
    """Keep the best box per label without joining nearby separate instances."""
    output = []
    for item in sorted(detections, key=lambda d: d.confidence or 0, reverse=True):
        if not any(item.label == other.label and box_iou(item.bbox, other.bbox) >= threshold
                   for other in output):
            output.append(item)
    return output


def run_photo(args):
    # Validate the input before loading weights or creating output directories.
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise ValueError(f"Could not read image: {args.image}")
    device = select_device(args.device)
    trained = YOLO(args.open_images_model)
    label_map = {name: name.lower() for name in trained.names.values()}
    names = {name.lower(): index for index, name in trained.names.items()}
    targets = parse_classes(args.classes) if args.classes else None
    targets = list(dict.fromkeys(ALIASES.get(t.lower(), t.lower()) for t in targets)) if targets else None
    class_ids = [names[t] for t in targets if t in names] if targets else None
    extra = [t for t in targets if t not in names] if targets else EXTRA_CLASSES
    detections = []
    # Photos have no latency budget: always scan the full image and every tile.
    # Use suppression rather than unions so separate instances remain separate.
    if class_ids is None or class_ids:
        detections.extend(predict_robust(trained, frame, args, device,
                                         class_ids=class_ids, label_map=label_map,
                                         merge=suppress_duplicates))
    if extra:
        prompted = YOLOE(args.model)
        prompted.set_classes(extra)
        detections.extend(predict_robust(prompted, frame, args, device,
                                         merge=suppress_duplicates))
    detections = suppress_duplicates(detections)
    height, width = frame.shape[:2]
    annotated = frame.copy()
    draw_detections(annotated, detections)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f"{args.image.stem}_annotated.jpg"
    json_path = args.output_dir / f"{args.image.stem}_detections.json"
    if not cv2.imwrite(str(image_path), annotated):
        raise RuntimeError(f"Could not write annotated image: {image_path}")
    json_path.write_text(json.dumps({
        "input": str(args.image.resolve()), "width": width, "height": height,
        "analyzed_at": datetime.now().astimezone().isoformat(),
        "bbox_format": "xyxy in original image pixels", "confidence_threshold": args.confidence,
        "models": [args.open_images_model] + ([args.model] if extra else []),
        "detections": [dict(instance_id=i, **asdict(d)) for i, d in enumerate(detections, 1)],
    }, indent=2), encoding="utf-8")
    print(f"Detected {len(detections)} instances. Scores are model confidence, not verified accuracy.")
    print(f"Annotated image: {image_path.resolve()}")
    print(f"All detections: {json_path.resolve()}")
