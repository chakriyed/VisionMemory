import base64
import re
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from . import config


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "object"


def decode_base64_image(data_url_or_b64: str) -> np.ndarray:
    """Decode a base64 (optionally data-URL prefixed) JPEG/PNG string to a BGR numpy array."""
    if "," in data_url_or_b64 and data_url_or_b64.strip().startswith("data:"):
        data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
    raw = base64.b64decode(data_url_or_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image data")
    return frame


def save_frame(frame: np.ndarray, subdir: str, filename: str) -> Path:
    out_dir = config.FRAMES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), frame)
    return out_path


def save_reference_image(frame: np.ndarray, object_slug: str) -> Path:
    out_path = config.REFERENCES_DIR / f"{object_slug}.jpg"
    cv2.imwrite(str(out_path), frame)
    return out_path


def frame_to_base64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Could not encode frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def file_to_base64(path: Path) -> Optional[str]:
    if not path or not Path(path).exists():
        return None
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2,
        )
    return annotated
