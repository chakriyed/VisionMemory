import logging
import threading
from typing import Any, Optional

import numpy as np

from . import config

logger = logging.getLogger("visionmemory.detector")


class ObjectDetector:
    """Wraps a YOLO-World open-vocabulary detector with built-in ByteTrack tracking.

    YOLO-World lets us detect arbitrary text-prompted classes (e.g. "headphones",
    "wallet", "usb drive") without any custom training, which regular COCO-trained
    YOLO can't do since those aren't COCO classes.
    """

    def __init__(self, model_name: str = config.YOLO_WORLD_MODEL):
        self._model_name = model_name
        self._model: Optional[Any] = None
        self._current_classes: list[str] = []
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        try:
            from ultralytics import YOLOWorld  # imported lazily: slow + needs weights

            logger.info("Loading YOLO-World model %s (first run downloads weights)...", self._model_name)
            self._model = YOLOWorld(self._model_name)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.exception("Failed to load YOLO-World model")

    @property
    def ready(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _set_classes(self, names: list[str]) -> None:
        if names and names != self._current_classes:
            self._model.set_classes(names)
            self._current_classes = list(names)

    def detect_and_track(self, frame: np.ndarray, class_names: list[str]) -> list[dict]:
        """Runs detection + tracking for the given frame, restricted to class_names.

        Returns a list of dicts: {track_id, class_name, confidence, bbox: [x1,y1,x2,y2]}
        """
        if not class_names:
            return []

        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError(f"Detector unavailable: {self._load_error}")

        with self._lock:
            self._set_classes(class_names)
            results = self._model.track(
                frame,
                persist=True,
                conf=config.DETECTION_CONF_THRESHOLD,
                verbose=False,
            )

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []

        names = result.names
        detections = []
        for box_xyxy, track_id, cls_idx, conf in zip(
            boxes.xyxy.tolist(), boxes.id.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
        ):
            detections.append(
                {
                    "track_id": int(track_id),
                    "class_name": names[int(cls_idx)],
                    "confidence": float(conf),
                    "bbox": [float(v) for v in box_xyxy],
                }
            )
        return detections


detector = ObjectDetector()
