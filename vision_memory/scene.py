"""Ground scene questions in measured, two-dimensional detection geometry."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
import math
from pathlib import Path


def load_scene(path: Path) -> dict:
    try:
        scene = json.loads(path.read_text(encoding="utf-8"))
        width, height = scene["width"], scene["height"]
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0
                   for v in (width, height)):
            raise ValueError("Invalid image dimensions")
        for item in scene["detections"]:
            box = item["bbox"]
            if (not isinstance(item["label"], str) or not item["label"].strip()
                    or len(box) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= width
                    or not 0 <= box[1] < box[3] <= height):
                raise ValueError("Invalid detection")
        return scene
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot read scene {path}: {error}") from error


def objects(scene: dict) -> list:
    counts = Counter()
    result = []
    for item in scene["detections"]:
        label = item["label"]
        counts[label] += 1
        box = [item["bbox"][i] / scene["width" if i % 2 == 0 else "height"] for i in range(4)]
        result.append({"id": f"{label} #{counts[label]}", "label": label,
                       "confidence": item.get("confidence"), "box": box,
                       "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]})
    return result


def position(center: list) -> str:
    x, y = center
    horizontal = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"
    vertical = "top" if y < 1 / 3 else "bottom" if y > 2 / 3 else "middle"
    return f"{vertical}-{horizontal}"


def spatial_relations(items: list, near_threshold: float = .08) -> list:
    relations = []
    for a, b in combinations(items, 2):
        ax1, ay1, ax2, ay2 = a["box"]
        bx1, by1, bx2, by2 = b["box"]
        dx = max(ax1 - bx2, bx1 - ax2, 0)
        dy = max(ay1 - by2, by1 - ay2, 0)
        gap = math.hypot(dx, dy) / math.sqrt(2)
        tags = []
        if ax2 <= bx1:
            tags.append("left of")
        elif bx2 <= ax1:
            tags.append("right of")
        if ay2 <= by1:
            tags.append("above")
        elif by2 <= ay1:
            tags.append("below")
        overlap = min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)
        if overlap:
            tags.append("boxes overlap")
        elif gap <= near_threshold:
            tags.append("next to")
        if tags:
            relations.append({"subject": a["id"], "object": b["id"],
                              "relations": tags, "edge_gap": round(gap, 4)})
    return relations


def compare_scenes(current: dict, previous: dict | None, fixed_camera: bool = False,
                   movement_threshold: float = .04) -> dict:
    if previous is None:
        return {"available": False, "reason": "Movement requires a previous scene."}
    now, before = defaultdict(list), defaultdict(list)
    for item in objects(current):
        now[item["label"]].append(item)
    for item in objects(previous):
        before[item["label"]].append(item)
    changes = []
    for label in sorted(now.keys() | before.keys()):
        old, new = before[label], now[label]
        if not old:
            changes.append({"label": label, "status": "newly detected", "count": len(new)})
        elif not new:
            changes.append({"label": label, "status": "not detected now; may be missed or occluded", "count": len(old)})
        elif len(old) != 1 or len(new) != 1:
            changes.append({"label": label, "status": "ambiguous identity: multiple instances",
                            "previous_count": len(old), "current_count": len(new)})
        else:
            delta = [new[0]["center"][i] - old[0]["center"][i] for i in (0, 1)]
            distance = math.hypot(*delta) / math.sqrt(2)
            changed = distance > movement_threshold
            changes.append({"label": label, "status": (
                "possibly moved" if fixed_camera else "image position changed; camera motion unknown"
            ) if changed else "no significant center change",
                "previous_center": old[0]["center"], "current_center": new[0]["center"],
                "from_position": position(old[0]["center"]),
                "to_position": position(new[0]["center"]),
                "normalized_displacement": round(distance, 4)})
    return {"available": True, "fixed_camera_asserted": fixed_camera,
            "movement_threshold": movement_threshold, "changes": changes,
            "caution": "Same-category matching is tentative, not physical identity verification. Detection jitter can look like motion."}


def scene_facts(current: dict, previous: dict | None = None, fixed_camera: bool = False) -> dict:
    items = objects(current)
    relations = spatial_relations(items)
    neighbors = {item["id"]: [] for item in items}
    for relation in relations:
        if "next to" in relation["relations"]:
            neighbors[relation["subject"]].append(relation["object"])
            neighbors[relation["object"]].append(relation["subject"])
    return {"source": current.get("input"), "analyzed_at": current.get("analyzed_at"),
            "next_to": neighbors,
            "objects": items, "spatial_relations": relations,
            "comparison": compare_scenes(current, previous, fixed_camera),
            "limitations": ["Relations describe the 2D image, not physical depth or contact.",
                           "Next to means non-overlapping boxes within 8% of the normalized diagonal.",
                           "Detections may be wrong or incomplete. Scores are not measured accuracy.",
                           "Object numbers only identify instances within this scene."]}
