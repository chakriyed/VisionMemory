"""High-resolution inference helpers for small objects."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .memory import Detection


def tile_regions(width: int, height: int, grid: int = 2) -> List[Tuple[int, int, int, int]]:
    """Return overlapping regions that cover the original frame."""
    if grid < 2:
        raise ValueError("Tile grid must be at least 2.")
    tile_width = min(width, int(width / grid * 1.18))
    tile_height = min(height, int(height / grid * 1.18))
    x_starts = np.linspace(0, width - tile_width, grid, dtype=int)
    y_starts = np.linspace(0, height - tile_height, grid, dtype=int)
    return [
        (int(x), int(y), int(x + tile_width), int(y + tile_height))
        for y in y_starts
        for x in x_starts
    ]


def brighten_if_dark(frame: np.ndarray, threshold: float = 85.0) -> np.ndarray:
    """Lift local contrast only for an underexposed camera frame."""
    sample = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    sample_lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
    if float(sample_lab[:, :, 0].mean()) >= threshold:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    a_area = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    b_area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(a_area + b_area - intersection, 1)


def box_overlap_smaller(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Measure whether one crop detected a substantial part of another box."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    a_area = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    b_area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(min(a_area, b_area), 1)


def merge_detections(detections: Iterable[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """Unify partial views of the same object from overlapping tiles."""
    ordered = sorted(detections, key=lambda item: item.confidence or 0, reverse=True)
    groups: List[List[Detection]] = []
    for item in ordered:
        matching_groups = [
            group
            for group in groups
            if item.label == group[0].label
            and any(
                box_iou(item.bbox, previous.bbox) >= iou_threshold
                or box_overlap_smaller(item.bbox, previous.bbox) >= 0.3
                for previous in group
            )
        ]
        if matching_groups:
            primary = matching_groups[0]
            primary.append(item)
            for other in matching_groups[1:]:
                primary.extend(other)
                groups.remove(other)
        else:
            groups.append([item])

    merged: List[Detection] = []
    for group in groups:
        best = group[0]
        boxes = [item.bbox for item in group]
        combined_box = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        merged.append(replace(best, bbox=combined_box))
    return merged


class FrameAssociator:
    """Assign stable IDs to merged boxes in robust inference mode."""

    def __init__(self, max_missing_frames: int = 8) -> None:
        self.max_missing_frames = max_missing_frames
        self.next_id = 1
        self.frame_number = 0
        self.tracks: Dict[int, Tuple[Detection, int]] = {}

    def update(self, detections: Sequence[Detection], width: int, height: int) -> List[Detection]:
        self.frame_number += 1
        self.tracks = {
            track_id: value
            for track_id, value in self.tracks.items()
            if self.frame_number - value[1] <= self.max_missing_frames
        }
        used_ids = set()
        output: List[Detection] = []
        diagonal = max(float(np.hypot(width, height)), 1.0)

        for item in detections:
            center_x = (item.bbox[0] + item.bbox[2]) / 2
            center_y = (item.bbox[1] + item.bbox[3]) / 2
            candidates = []
            for track_id, (previous, _) in self.tracks.items():
                if track_id in used_ids or previous.label != item.label:
                    continue
                prev_x = (previous.bbox[0] + previous.bbox[2]) / 2
                prev_y = (previous.bbox[1] + previous.bbox[3]) / 2
                distance = np.hypot(center_x - prev_x, center_y - prev_y) / diagonal
                overlap = box_iou(item.bbox, previous.bbox)
                if overlap >= 0.1 or distance <= 0.035:
                    candidates.append((overlap - distance, track_id))

            if candidates:
                track_id = max(candidates)[1]
            else:
                track_id = self.next_id
                self.next_id += 1
            used_ids.add(track_id)
            tracked = replace(item, track_id=track_id)
            self.tracks[track_id] = (tracked, self.frame_number)
            output.append(tracked)
        return output

    def recent(self, current: Sequence[Detection], max_age: int) -> List[Detection]:
        """Show unmatched boxes briefly, but do not treat them as observations."""
        if max_age < 1:
            return []
        current_ids = {item.track_id for item in current}
        return [
            replace(item, source="recent")
            for track_id, (item, last_frame) in self.tracks.items()
            if track_id not in current_ids
            and 0 < self.frame_number - last_frame <= max_age
        ]
