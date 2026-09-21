import sqlite3
from typing import Optional

from . import config, storage


def find_referenced_object(question: str, objects: list[sqlite3.Row]) -> Optional[sqlite3.Row]:
    """Best-effort match of which registered object a free-text question refers to."""
    q = question.lower()
    # Prefer the longest matching name so "usb drive" beats a hypothetical "drive".
    matches = [obj for obj in objects if obj["name"] in q]
    if not matches:
        return None
    return max(matches, key=lambda o: len(o["name"]))


def _describe_event(event: sqlite3.Row) -> str:
    kind = "last detected" if event["event_type"] == "seen" else "disappeared from view"
    return f"{event['timestamp']} — {kind} (confidence {event['confidence']:.2f})" if event["confidence"] is not None else f"{event['timestamp']} — {kind}"


def answer_question(question: str, object_row: sqlite3.Row, events: list[sqlite3.Row]) -> dict:
    if not events:
        return {
            "answer": f"I have no recorded sightings of '{object_row['display_name']}' yet.",
            "evidence": [],
        }

    if not config.ANTHROPIC_API_KEY:
        return {
            "answer": (
                "ANTHROPIC_API_KEY is not set on the server, so I can't call Claude to "
                "interpret the stored frames. Set the ANTHROPIC_API_KEY environment "
                f"variable and restart the backend. Meanwhile, here is the raw event "
                f"history I found for '{object_row['display_name']}':\n"
                + "\n".join(f"- {_describe_event(e)}" for e in events)
            ),
            "evidence": _events_to_evidence(events),
        }

    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    evidence_events = events[: config.MAX_EVIDENCE_FRAMES]
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"You are the memory module of a visual-memory assistant. A user is asking about "
                f"an object they registered as '{object_row['display_name']}'. Below are the most "
                f"recent camera evidence frames for this object, newest first, each with its "
                f"timestamp and event type (a 'seen' event is when the object was detected; a "
                f"'disappeared' event means it was last tracked there before going out of view). "
                f"Use ONLY this evidence to answer. Be specific about location (describe where in "
                f"the frame / what it's near or on) and time. If the evidence doesn't answer the "
                f"question, say so plainly.\n\nUser question: {question}"
            ),
        }
    ]

    for event in evidence_events:
        b64 = storage.file_to_base64(event["frame_path"]) if event["frame_path"] else None
        label = "Last detected" if event["event_type"] == "seen" else "Disappeared from view"
        content.append(
            {
                "type": "text",
                "text": f"\nEvidence frame — {label} at {event['timestamp']}:",
            }
        )
        if b64:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                }
            )

    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": content}],
    )
    answer_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )

    return {"answer": answer_text, "evidence": _events_to_evidence(evidence_events)}


def _events_to_evidence(events: list[sqlite3.Row]) -> list[dict]:
    return [
        {
            "event_type": e["event_type"],
            "timestamp": e["timestamp"],
            "confidence": e["confidence"],
            "frame_path": e["frame_path"],
        }
        for e in events
    ]
