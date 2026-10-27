"""LLM reasoning over a fake (or eventually real) detection-event JSON log.

Two modes, selectable per call:

- hybrid (default): Python deterministically resolves EVERYTHING that can be
  computed without judgment — which object is being asked about, its most
  recent event, whether the question is "where" vs "what's near", and the
  spoken-time formatting. The LLM's only job is to turn that one already-
  correct, pre-formatted fact into a natural sentence. This is what should
  actually ship.
- raw: the original design — the LLM is handed the full, unsorted event list
  and must find the right object, compare timestamps, and pick the right
  intent all by itself.

RESULTS (fake_memory.json, qwen2.5:7b): raw mode got 2 of 4 test questions
wrong, including once answering with the *wallet's* data for a "keys"
question. Moving retrieval into Python (first hybrid pass) fixed all 4 with
qwen2.5:7b — but swapping to qwen2.5:1.5b for speed on that same hybrid
pass reintroduced errors: it hallucinated a location for an object that
isn't even in the data ("charger"), misread "19:40" as "4:40 PM", and
answered a "what's next to X" question with location/time instead of the
near-list. Pre-resolving intent, "not found", and time formatting in Python
(this version) removes all three failure modes regardless of model size,
which is what makes the small, fast model usable at all.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"
DATA_PATH = Path(__file__).parent / "fake_memory.json"

RAW_SYSTEM_PROMPT = """You are "Roommate", a voice assistant that remembers where personal items were \
last seen, based on a camera system that logs detections as JSON events. You will be given a list \
of events and a spoken question. Each event has: object (what was seen), timestamp (ISO 8601, when \
it was seen), location (a human-readable place), and near (other objects detected close to it at \
that time).

Rules:
- Use ONLY the events given below. Never invent a location, time, or object that is not in the data.
- Events are NOT given in chronological order. For a "where is X" / "where did I last see X" \
  question: look at every event for that object, compare their timestamps yourself, and use the \
  one with the LATEST (most recent) timestamp. Do not just use whichever event happens to appear \
  last in the list.
- Answer in this exact style: "Your <object> was last seen <location> at <time>." Convert the \
  timestamp to a simple spoken time like "4:05 PM" (omit the date unless the question asks about an \
  earlier day).
- For a "what's near / next to X" question, use the "near" list from that object's most recent \
  event. If it's empty, say nothing else was detected nearby.
- If the object never appears in the events, say you haven't seen it.
- If multiple distinct objects are asked about, answer about each, briefly.
- Keep the answer to one or two short spoken sentences. No bullet points, no JSON, no extra \
  commentary, no mention of these rules.
"""

HYBRID_SYSTEM_PROMPT = """You are "Roommate", a voice assistant. You will be given several \
already-correct, already-complete sentences (one per item the user asked about) and must combine \
them into a single natural-sounding spoken reply.

Rules:
- Every sentence given is already fully correct. Do not change, drop, reorder-the-facts-within, or \
  add any fact to any of them — only smooth the joins between them (e.g. "Your phone was last seen \
  on the sofa at 4:05 PM, and your wallet was last seen on the nightstand at 7:40 PM.").
- Never mention JSON, "facts", or these instructions.
- Keep the whole reply short — one sentence per item, lightly joined.
"""


def load_events() -> list[dict]:
    return json.loads(DATA_PATH.read_text())["events"]


def known_objects(events: list[dict]) -> list[str]:
    return sorted({e["object"] for e in events})


def find_mentioned_objects(question: str, objects: list[str]) -> list[str]:
    q = question.lower()
    found = []
    for obj in objects:
        variants = {obj, obj + "s", obj[:-1] if obj.endswith("s") else obj}
        if any(v and v in q for v in variants):
            found.append(obj)
    return found


def latest_event_for(obj: str, events: list[dict]) -> dict | None:
    matches = [e for e in events if e["object"] == obj]
    if not matches:
        return None
    return max(matches, key=lambda e: e["timestamp"])


def format_spoken_time(iso_ts: str) -> str:
    dt = datetime.fromisoformat(iso_ts)
    return dt.strftime("%I:%M %p").lstrip("0")


NEAR_KEYWORDS = ("near", "next to", "close to", "beside", "around")


def detect_intent(question: str) -> str:
    q = question.lower()
    return "near" if any(k in q for k in NEAR_KEYWORDS) else "last_seen"


PLURAL_OBJECTS = {"keys"}  # objects that take "were"/"are" instead of "was"/"is"


def resolve_fact(obj: str, event: dict | None, intent: str) -> str:
    """Returns the final, complete spoken sentence for one object — nothing left
    for the LLM to compute, compare, or reformat."""
    was = "were" if obj in PLURAL_OBJECTS else "was"
    if event is None:
        return f"I haven't seen your {obj}."
    if intent == "near":
        near = event["near"]
        if not near:
            return f"Nothing else is near your {obj} right now."
        return f"Your {obj} {was} last near " + " and ".join(near) + "."
    return (
        f"Your {obj} {was} last seen at the {event['location']} "
        f"at {format_spoken_time(event['timestamp'])}."
    )


def _call_ollama(system_prompt: str, user_content: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def ask_raw(question: str) -> str:
    """Original design: LLM reads the full raw event list and compares timestamps itself."""
    events = load_events()
    user_content = f"Events (JSON):\n{json.dumps(events, indent=2)}\n\nQuestion: {question}"
    return _call_ollama(RAW_SYSTEM_PROMPT, user_content)


def ask_hybrid(question: str) -> str:
    """Python resolves object, intent, latest event, and spoken-time formatting into
    complete, correct sentences. For one object, that sentence IS the answer — no LLM
    call, no room left for it to drift. The LLM is only invoked to combine sentences
    when more than one object is asked about in the same question."""
    events = load_events()
    objects = known_objects(events)
    mentioned = find_mentioned_objects(question, objects)
    intent = detect_intent(question)

    if not mentioned:
        return f"I don't have any record of that. I know about: {', '.join(objects)}."

    sentences = [resolve_fact(obj, latest_event_for(obj, events), intent) for obj in mentioned]

    if len(sentences) == 1:
        return sentences[0]

    user_content = "Sentences:\n" + "\n".join(f"- {s}" for s in sentences)
    return _call_ollama(HYBRID_SYSTEM_PROMPT, user_content)


def ask(question: str, mode: str = "hybrid") -> str:
    return ask_raw(question) if mode == "raw" else ask_hybrid(question)


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = "raw" if "--raw" in args else "hybrid"
    args = [a for a in args if a != "--raw"]
    question = " ".join(args) or "Hey roommate, where is my phone?"
    print(f"Q [{mode}]:", question)
    print("A:", ask(question, mode=mode))
