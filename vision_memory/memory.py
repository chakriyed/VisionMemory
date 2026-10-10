from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: Optional[float]
    bbox: BBox
    track_id: Optional[int]
    source: str = "model"


@dataclass
class LastSeen:
    last_seen_at: str
    confidence: Optional[float]
    track_id: Optional[int]
    bbox: BBox
    frame_position: str
    evidence_path: Optional[str]
    source: str = "model"


def frame_position(bbox: BBox, frame_width: int, frame_height: int) -> str:
    """Return a human-friendly 3x3 position for a bounding box center."""
    x1, y1, x2, y2 = bbox
    x = ((x1 + x2) / 2) / max(frame_width, 1)
    y = ((y1 + y2) / 2) / max(frame_height, 1)

    horizontal = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"
    vertical = "top" if y < 1 / 3 else "bottom" if y > 2 / 3 else "middle"
    return horizontal if vertical == "middle" else f"{vertical}-{horizontal}"


def center_distance(first: BBox, second: BBox, width: int, height: int) -> float:
    """Calculate normalized distance between two bounding-box centers."""
    ax = (first[0] + first[2]) / 2
    ay = (first[1] + first[3]) / 2
    bx = (second[0] + second[2]) / 2
    by = (second[1] + second[3]) / 2
    diagonal = max(math.hypot(width, height), 1)
    return math.hypot(ax - bx, ay - by) / diagonal


class MemoryStore:
    """Maintain and persist the best current observation for each class."""

    def __init__(
        self,
        data_dir: Path,
        evidence_interval_seconds: float = 5.0,
        movement_threshold: float = 0.08,
    ) -> None:
        self.data_dir = data_dir
        self.evidence_dir = data_dir / "evidence"
        self.state_path = data_dir / "state.json"
        self.evidence_interval_seconds = evidence_interval_seconds
        self.movement_threshold = movement_threshold
        self.objects: Dict[str, LastSeen] = {}
        self._last_evidence_time: Dict[str, datetime] = {}
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            for label, value in payload.get("objects", {}).items():
                value["bbox"] = tuple(value["bbox"])
                self.objects[label] = LastSeen(**value)
        except (OSError, ValueError, TypeError, KeyError):
            # A corrupt prior state should not prevent the camera from starting.
            self.objects = {}

    def update(
        self,
        detections: Iterable[Detection],
        frame: np.ndarray,
        observed_at: datetime,
    ) -> None:
        height, width = frame.shape[:2]
        # Keep only the highest-confidence detection per label for "last seen".
        best: Dict[str, Detection] = {}
        for detection in detections:
            current = best.get(detection.label)
            if current is None or (detection.confidence or 0) > (current.confidence or 0):
                best[detection.label] = detection

        for label, detection in best.items():
            previous = self.objects.get(label)
            last_evidence = self._last_evidence_time.get(label)
            elapsed = (
                (observed_at - last_evidence).total_seconds()
                if last_evidence is not None
                else float("inf")
            )
            moved = (
                previous is None
                or center_distance(previous.bbox, detection.bbox, width, height)
                >= self.movement_threshold
            )
            new_track = (
                previous is not None
                and detection.track_id is not None
                and detection.track_id != previous.track_id
            )
            should_save_evidence = previous is None or moved or new_track or (
                elapsed >= self.evidence_interval_seconds
            )

            evidence_path = previous.evidence_path if previous else None
            if should_save_evidence:
                evidence_path = self._save_evidence(label, frame, observed_at)
                self._last_evidence_time[label] = observed_at

            self.objects[label] = LastSeen(
                last_seen_at=observed_at.isoformat(),
                confidence=(
                    round(detection.confidence, 4)
                    if detection.confidence is not None
                    else None
                ),
                track_id=detection.track_id,
                bbox=detection.bbox,
                frame_position=frame_position(detection.bbox, width, height),
                evidence_path=evidence_path,
                source=detection.source,
            )

        if best:
            self._persist()

    def _save_evidence(
        self, label: str, frame: np.ndarray, observed_at: datetime
    ) -> str:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "_".join(label.lower().split())
        filename = f"{safe_label}_{observed_at.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        path = self.evidence_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)

    def _persist(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now().astimezone().isoformat(),
            "objects": {label: asdict(value) for label, value in self.objects.items()},
        }
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def summary(self) -> str:
        if not self.objects:
            return "No objects have been seen yet."
        lines = ["Last seen objects:"]
        for label, item in sorted(self.objects.items()):
            detail = (
                "manually selected"
                if item.source == "manual"
                else f"confidence {item.confidence:.0%}, track {item.track_id}"
            )
            lines.append(
                f"- {label}: {item.frame_position} at {item.last_seen_at} "
                f"({detail})"
            )
        return "\n".join(lines)
