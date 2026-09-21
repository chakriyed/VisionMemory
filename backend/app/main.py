import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config, db, storage, vlm
from .detector import detector
from .tracking import memory_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visionmemory")

app = FastAPI(title="VisionMemory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@app.on_event("startup")
def on_startup() -> None:
    db.init_db()


def to_url(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    try:
        rel = Path(path_str).resolve().relative_to(config.DATA_DIR.resolve())
    except ValueError:
        return None
    return f"/data/{rel.as_posix()}"


class RegisterObjectRequest(BaseModel):
    name: str
    image: str  # base64 (optionally data-URL prefixed)


class FrameRequest(BaseModel):
    image: str


class QueryRequest(BaseModel):
    question: str


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "detector_ready": detector.ready,
        "detector_error": detector.load_error,
        "anthropic_configured": bool(config.ANTHROPIC_API_KEY),
    }


@app.post("/api/objects")
def register_object(req: RegisterObjectRequest) -> dict:
    display_name = req.name.strip()
    if not display_name:
        raise HTTPException(400, "Object name is required")
    normalized = display_name.lower()

    if db.get_object_by_name(normalized) is not None:
        raise HTTPException(409, f"An object named '{display_name}' is already registered")

    try:
        frame = storage.decode_base64_image(req.image)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc

    slug = storage.slugify(normalized)
    ref_path = storage.save_reference_image(frame, slug)

    object_id = db.create_object(normalized, display_name, str(ref_path))
    return {"id": object_id, "name": normalized, "display_name": display_name}


@app.get("/api/objects")
def list_objects() -> list[dict]:
    objects = db.list_objects()
    result = []
    for obj in objects:
        last_event = db.get_last_event(obj["id"])
        result.append(
            {
                "id": obj["id"],
                "name": obj["name"],
                "display_name": obj["display_name"],
                "reference_image_url": to_url(obj["reference_image_path"]),
                "created_at": obj["created_at"],
                "last_event": (
                    {
                        "event_type": last_event["event_type"],
                        "timestamp": last_event["timestamp"],
                    }
                    if last_event
                    else None
                ),
            }
        )
    return result


@app.delete("/api/objects/{object_id}")
def delete_object(object_id: int) -> dict:
    obj = db.get_object_by_id(object_id)
    if obj is None:
        raise HTTPException(404, "Object not found")
    memory_engine.forget_object(obj["name"])
    db.delete_object(object_id)
    return {"deleted": object_id}


@app.get("/api/objects/{object_id}/events")
def object_events(object_id: int, limit: int = 20) -> list[dict]:
    obj = db.get_object_by_id(object_id)
    if obj is None:
        raise HTTPException(404, "Object not found")
    events = db.get_recent_events(object_id, limit=limit)
    return [
        {
            "event_type": e["event_type"],
            "timestamp": e["timestamp"],
            "confidence": e["confidence"],
            "bbox": [e["bbox_x1"], e["bbox_y1"], e["bbox_x2"], e["bbox_y2"]],
            "frame_url": to_url(e["frame_path"]),
        }
        for e in events
    ]


@app.post("/api/frame")
def process_frame(req: FrameRequest) -> dict:
    try:
        frame = storage.decode_base64_image(req.image)
    except ValueError as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc

    if not detector.ready:
        raise HTTPException(503, f"Detector not ready: {detector.load_error or 'loading...'}")

    try:
        detections = memory_engine.process_frame(frame)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    return {
        "detections": [
            {
                "track_id": d["track_id"],
                "class_name": d["class_name"],
                "confidence": d["confidence"],
                "bbox": d["bbox"],
            }
            for d in detections
        ]
    }


@app.post("/api/query")
def query(req: QueryRequest) -> dict:
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is required")

    objects = db.list_objects()
    if not objects:
        return {
            "answer": "No objects have been registered yet, so I have nothing to look up.",
            "object": None,
            "evidence": [],
        }

    target = vlm.find_referenced_object(question, objects)
    if target is None:
        names = ", ".join(o["display_name"] for o in objects)
        return {
            "answer": f"I couldn't tell which registered object you mean. Registered objects: {names}.",
            "object": None,
            "evidence": [],
        }

    events = db.get_recent_events(target["id"], limit=config.MAX_EVIDENCE_FRAMES)
    result = vlm.answer_question(question, target, events)
    evidence = [
        {**e, "frame_url": to_url(e["frame_path"])} for e in result["evidence"]
    ]
    for e in evidence:
        e.pop("frame_path", None)

    return {
        "answer": result["answer"],
        "object": {"id": target["id"], "display_name": target["display_name"]},
        "evidence": evidence,
    }


app.mount("/data", StaticFiles(directory=str(config.DATA_DIR)), name="data")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))
