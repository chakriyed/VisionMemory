import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = Path(os.environ.get("VISIONMEMORY_DATA_DIR", PROJECT_ROOT / "data")).resolve()
FRAMES_DIR = DATA_DIR / "frames"
REFERENCES_DIR = DATA_DIR / "references"
DB_PATH = DATA_DIR / "visionmemory.db"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

# Detection
YOLO_WORLD_MODEL = os.environ.get("VISIONMEMORY_YOLO_MODEL", "yolov8s-world.pt")
DETECTION_CONF_THRESHOLD = float(os.environ.get("VISIONMEMORY_CONF_THRESHOLD", "0.15"))

# How often (seconds) we persist a new "seen" event + frame for the same
# object track, so we don't flood storage with a frame every second.
MIN_SECONDS_BETWEEN_EVENTS = float(os.environ.get("VISIONMEMORY_EVENT_INTERVAL", "3.0"))

# If a previously-tracked object isn't detected for this long, we log a
# "disappeared" event using its last known frame/location.
DISAPPEARANCE_TIMEOUT_SECONDS = float(os.environ.get("VISIONMEMORY_DISAPPEAR_TIMEOUT", "5.0"))

# Max evidence frames sent to the VLM per query.
MAX_EVIDENCE_FRAMES = int(os.environ.get("VISIONMEMORY_MAX_EVIDENCE", "4"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("VISIONMEMORY_ANTHROPIC_MODEL", "claude-sonnet-5")
