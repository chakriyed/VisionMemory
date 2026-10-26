from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vision_memory.memory import Detection, MemoryStore, center_distance, frame_position


def test_frame_position() -> None:
    assert frame_position((0, 0, 20, 20), 300, 300) == "top-left"
    assert frame_position((140, 140, 160, 160), 300, 300) == "center"
    assert frame_position((280, 280, 300, 300), 300, 300) == "bottom-right"


def test_center_distance() -> None:
    assert center_distance((0, 0, 10, 10), (0, 0, 10, 10), 100, 100) == 0
    assert center_distance((0, 0, 10, 10), (90, 90, 100, 100), 100, 100) > 0.8


def test_memory_store_persists_highest_confidence(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    timestamp = datetime(2026, 10, 10, 8, 30, tzinfo=timezone.utc)
    store.update(
        [
            Detection("wallet", 0.4, (0, 0, 20, 20), 1),
            Detection("wallet", 0.9, (40, 40, 70, 70), 2),
        ],
        frame,
        timestamp,
    )

    assert store.objects["wallet"].confidence == 0.9
    assert store.objects["wallet"].track_id == 2
    assert (tmp_path / "state.json").exists()
    assert Path(store.objects["wallet"].evidence_path).exists()

    reloaded = MemoryStore(tmp_path)
    assert reloaded.objects["wallet"].bbox == (40, 40, 70, 70)


def test_manual_pen_does_not_claim_model_confidence(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    store.update(
        [Detection("pen", None, (10, 10, 60, 20), None, source="manual")],
        frame,
        datetime(2026, 10, 10, 8, 30, tzinfo=timezone.utc),
    )

    reloaded = MemoryStore(tmp_path)
    assert reloaded.objects["pen"].source == "manual"
    assert reloaded.objects["pen"].confidence is None
    assert "manually selected" in reloaded.summary()
