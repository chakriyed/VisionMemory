import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config, db, storage
from .detector import detector


@dataclass
class TrackState:
    last_seen_monotonic: float
    last_bbox: tuple[float, float, float, float]
    last_confidence: float
    last_frame_path: Optional[str]
    last_event_monotonic: float = 0.0
    disappeared_logged: bool = True  # start True so we don't log a spurious disappearance


@dataclass
class ObjectMemoryState:
    tracks: dict[int, TrackState] = field(default_factory=dict)


class MemoryEngine:
    """Maintains in-process tracking state across successive /api/frame calls and
    writes throttled 'seen' / 'disappeared' events to the database."""

    def __init__(self) -> None:
        self._state: dict[str, ObjectMemoryState] = {}
        self._lock = threading.Lock()

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        objects = db.list_objects()
        if not objects:
            return []

        name_to_object = {obj["name"]: obj for obj in objects}
        class_names = list(name_to_object.keys())

        detections = detector.detect_and_track(frame, class_names)
        now = time.monotonic()

        with self._lock:
            seen_track_ids_by_name: dict[str, set[int]] = {}

            for det in detections:
                obj_name = det["class_name"]
                obj_row = name_to_object.get(obj_name)
                if obj_row is None:
                    continue  # detector returned a class we didn't ask about

                mem = self._state.setdefault(obj_name, ObjectMemoryState())
                seen_track_ids_by_name.setdefault(obj_name, set()).add(det["track_id"])

                track = mem.tracks.get(det["track_id"])
                should_log_seen = track is None or (
                    now - track.last_event_monotonic >= config.MIN_SECONDS_BETWEEN_EVENTS
                )

                frame_path = track.last_frame_path if track else None
                if should_log_seen:
                    annotated = storage.draw_detections(frame, [det])
                    saved = storage.save_frame(
                        annotated,
                        storage.slugify(obj_name),
                        f"{int(time.time() * 1000)}_seen.jpg",
                    )
                    frame_path = str(saved)
                    db.create_event(
                        object_id=obj_row["id"],
                        track_id=det["track_id"],
                        event_type="seen",
                        confidence=det["confidence"],
                        bbox=tuple(det["bbox"]),
                        frame_path=frame_path,
                    )

                mem.tracks[det["track_id"]] = TrackState(
                    last_seen_monotonic=now,
                    last_bbox=tuple(det["bbox"]),
                    last_confidence=det["confidence"],
                    last_frame_path=frame_path,
                    last_event_monotonic=now if should_log_seen else track.last_event_monotonic,
                    disappeared_logged=False,
                )

            # Check for disappearances: tracks we knew about but didn't see this frame.
            for obj_name, mem in self._state.items():
                obj_row = name_to_object.get(obj_name)
                if obj_row is None:
                    continue
                still_seen = seen_track_ids_by_name.get(obj_name, set())
                for track_id, track in mem.tracks.items():
                    if track_id in still_seen or track.disappeared_logged:
                        continue
                    if now - track.last_seen_monotonic >= config.DISAPPEARANCE_TIMEOUT_SECONDS:
                        db.create_event(
                            object_id=obj_row["id"],
                            track_id=track_id,
                            event_type="disappeared",
                            confidence=track.last_confidence,
                            bbox=track.last_bbox,
                            frame_path=track.last_frame_path,
                        )
                        track.disappeared_logged = True

        return detections

    def forget_object(self, object_name: str) -> None:
        with self._lock:
            self._state.pop(object_name, None)


memory_engine = MemoryEngine()
