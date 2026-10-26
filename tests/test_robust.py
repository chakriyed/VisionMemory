import numpy as np
from types import SimpleNamespace

from vision_memory import app
from vision_memory.memory import Detection
from vision_memory.robust import FrameAssociator, brighten_if_dark, merge_detections, tile_regions


def test_tiles_cover_edges_and_overlap() -> None:
    regions = tile_regions(1280, 720, grid=3)
    assert len(regions) == 9
    assert regions[0][:2] == (0, 0)
    assert regions[-1][2:] == (1280, 720)
    assert regions[0][2] > regions[1][0]


def test_brightening_only_changes_dark_frames() -> None:
    bright = np.full((64, 64, 3), 180, dtype=np.uint8)
    assert brighten_if_dark(bright) is bright
    dark = np.full((64, 64, 3), 20, dtype=np.uint8)
    assert brighten_if_dark(dark).shape == dark.shape


def test_merge_and_associate_detections() -> None:
    duplicate = [
        Detection("pen", 0.8, (100, 100, 200, 130), None),
        Detection("pen", 0.6, (102, 101, 202, 131), None),
    ]
    merged = merge_detections(duplicate)
    assert len(merged) == 1

    tracker = FrameAssociator()
    first = tracker.update(merged, 1280, 720)
    next_frame = tracker.update(
        [Detection("pen", 0.7, (105, 103, 205, 133), None)], 1280, 720
    )
    assert next_frame[0].track_id == first[0].track_id


def test_merge_partial_pen_boxes_from_tiles() -> None:
    partial_boxes = [
        Detection("pen", 0.8, (825, 576, 1510, 1109), None),
        Detection("pen", 0.75, (1051, 558, 1887, 1028), None),
        Detection("pen", 0.7, (1186, 420, 1926, 802), None),
    ]
    merged = merge_detections(partial_boxes)
    assert len(merged) == 1
    assert merged[0].bbox == (825, 420, 1926, 1109)


def test_recent_box_is_not_a_fresh_detection() -> None:
    tracker = FrameAssociator()
    first = tracker.update(
        [Detection("pen", 0.8, (100, 100, 200, 130), None)], 1280, 720
    )
    assert tracker.recent(first, 3) == []
    tracker.update([], 1280, 720)
    recent = tracker.recent([], 3)
    assert len(recent) == 1
    assert recent[0].track_id == first[0].track_id
    assert recent[0].source == "recent"
    for _ in range(3):
        tracker.update([], 1280, 720)
    assert tracker.recent([], 3) == []


def test_scheduled_tiles_use_larger_inference_size(monkeypatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.sizes = []

        def predict(self, frame, **options):
            self.sizes.append(options["imgsz"])
            return [object()]

    monkeypatch.setattr(app, "detections_from_result", lambda *args: [])
    model = FakeModel()
    args = SimpleNamespace(
        confidence=0.2, image_size=640, tile_image_size=960, tile_grid=2
    )
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)
    assert app.predict_robust(model, frame, args, "cpu", scan_tiles=False) == []
    assert model.sizes == [640]
    assert app.predict_robust(model, frame, args, "cpu", scan_tiles=True) == []
    assert model.sizes == [640, 640, 960, 960, 960, 960]
