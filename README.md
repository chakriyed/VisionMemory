# VisionMemory

A local webcam prototype that detects and tracks everyday objects and remembers
where they were last seen in the camera frame.

The first milestone tracks these classes by default:

- pen
- wristwatch
- set of keys
- wallet

Detections are shown live with persistent tracker IDs. The latest observation
for each object is written to `data/state.json`, and supporting frames are saved
under `data/evidence/`.

Pen and watch use an Open Images V7 detector trained on those categories. Keys
and wallets use the open-vocabulary detector. Running all four categories uses
both models and can be slower than testing one category at a time.

## Requirements

- macOS, Linux, or Windows with a webcam
- Python 3.9 or newer
- Internet access during the first run to download model weights and the text
  encoder

Apple Silicon Macs automatically use Metal (MPS) when it is available.

## Setup

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

For automatic pen detection with the built-in laptop camera (`--camera 0`):

```bash
python -m vision_memory --camera 0 --classes pen --robust \
  --tile-grid 2 --tile-interval 4 --image-size 640 \
  --tile-image-size 960 --confidence 0.15
```

macOS will ask for camera permission the first time. Allow camera access for
Terminal or Codex, whichever launches the process.

The program prints the actual resolution delivered by the laptop camera.
`--robust` runs a 640-pixel full-image pass every frame and scans overlapping
crops at 960 pixels every four frames by default. It adjusts dark crops when
needed. This helps smaller objects without paying for every crop on every frame. A box
labelled `recent, unconfirmed` is the last detected location between scans;
it does not update the saved last-seen time. For a farther pen, try
`--tile-grid 3 --tile-image-size 1280 --tile-interval 2` if the 2×2 setting misses
it. This increases detail and scan frequency, but reduces FPS. Set
`--tile-interval 1` to scan all crops on every frame.

To test the four initial categories together, remove `--classes pen`. This
loads two models and runs more slowly.

Controls:

- `q` or Escape: quit
- `s`: print the current last-seen summary
- `c`: save an unannotated camera frame under `data/diagnostics/` for debugging

Useful options:

```bash
python -m vision_memory --camera 0 --confidence 0.20
python -m vision_memory --classes "pen,wristwatch,set of keys,wallet,phone"
python -m vision_memory --device cpu
```

The laptop camera remains the only camera needed for this milestone. If a pen
at 2–3 metres is only a few pixels wide in the raw frame, software cannot
recover the missing detail. Press `c` at your intended distance, angle, and
lighting to save examples for evaluation and future fine-tuning.

### Testing a pen

The trained Open Images model downloads on its first use. Put a pen on a plain,
contrasting surface, fill a noticeable part of the image with it, and keep it
still for a few seconds. Lower confidence can also produce false positives, so
check the box before trusting a saved last-seen record.

## Output

`data/state.json` contains one last-seen entry per configured object:

```json
{
  "objects": {
    "wallet": {
      "last_seen_at": "2026-10-10T12:34:56.789012+05:30",
      "confidence": 0.82,
      "track_id": 4,
      "bbox": [100, 120, 260, 310],
      "frame_position": "center",
      "evidence_path": "data/evidence/wallet_20261010_123456_789012.jpg"
    }
  }
}
```

## Current limitation

This version recognizes categories, not a particular person's wallet or watch.
The next milestone is object registration: capturing several reference views
and matching detections against those references.

## Detect objects in a photo

Photo mode scans all 601 categories in the Open Images detector, plus text
prompts for keys, wallets, pencils, charging cables, phone chargers, and earbuds.
It runs a full-image pass and overlapping crop passes on every photo. No camera
or graphical window is required.

```bash
python -m vision_memory --image /path/to/items.jpg --model yoloe-26s-seg.pt \
  --image-size 960 --tile-image-size 1280 --confidence 0.25
```

Results are saved in `data/image_results/`: an annotated JPEG with labels and
boxes, and JSON containing every detected instance (including repeated items of
the same category), confidence scores, and bounding boxes in original pixels.
Use `--output-dir` to choose another destination. Results with the same input
stem overwrite earlier results in that directory. Photo mode leaves webcam
last-seen memory untouched.

To focus on particular objects or add a category outside the trained vocabulary:

```bash
python -m vision_memory --image /path/to/items.jpg \
  --classes "pen,wristwatch,phone,wallet,set of keys" --model yoloe-26s-seg.pt
```

Trained categories use the Open Images model; other names use text prompts.
The larger local YOLOE small model is optional; it consumes more compute than
the default nano model. Broad category coverage does not guarantee that every
item is detected or labeled correctly. Tiny, occluded, unusual, and ambiguous
objects may be missed or misclassified. Scores are model confidence, not
measured accuracy. Lowering the threshold may recover objects but adds false
positives. Evaluate representative photos with manually checked labels and boxes
before relying on results; photo accuracy has not yet been benchmarked.

## Ask questions about objects with a local LLM

The `chat` command reads saved photo detections and computes spatial relations
before sending those facts to Ollama. It supports interactive follow-up questions
and does not load the detection models again. It uses Ollama's `/api/chat`
endpoint (https://docs.ollama.com/api/chat); no additional Python package is needed.
Images are not sent to the LLM. By default, scene labels and coordinates go only
to Ollama on localhost; changing `--llm-host` sends those facts to that host.

Install/start Ollama, then download a model once:

```bash
ollama pull qwen2.5:7b
# If Ollama is not already running, start it in another terminal:
ollama serve
```

Chat about the existing photo result:

```bash
.venv/bin/python -m vision_memory chat \
  --scene data/image_results/image_fc2abec0_detections.json
```

Example questions: “What is next to the wallet?”, “What is to the left of the
phone?”, “How many pens are detected?” and “Where is pen #2?” Type `quit` to exit.
Use `--llm-model NAME` for another installed model. Chat reads a fixed scene at
startup; it does not monitor the camera or refresh files during the conversation.

For movement, photograph the scene twice with the same camera position and
framing. Use distinct filenames, such as `before.jpg` and `after.jpg`, and run
photo detection on each. Then compare the resulting files:

```bash
.venv/bin/python -m vision_memory chat \
  --scene data/image_results/after_detections.json \
  --previous data/image_results/before_detections.json \
  --fixed-camera --ask "Which items may have moved?"
```

Only use `--fixed-camera` when both photos have the same framing, zoom, and
camera position. Otherwise, the assistant reports changed image positions,
which could result from camera motion. A displacement must exceed 4% of the
normalized image diagonal to count as a significant center change. A matching
category appearing once in each photo is only a tentative identity match. When
multiple instances share a category, the comparison reports ambiguous identity
rather than claiming to know which physical item moved. A missing detection
may be an occlusion or a detector miss; it does not prove removal.

Spatial relations describe the image plane: left/right, above/below, overlapping
boxes, and “next to” (non-overlapping boxes separated by at most 8% of the
normalized image diagonal). They do not establish depth, physical contact, or
support. Scene IDs such as `pen #2` are local to one photo. LLM answers remain
subject to model errors; inspect the measured evidence with:

```bash
.venv/bin/python -m vision_memory chat \
  --scene data/image_results/after_detections.json \
  --previous data/image_results/before_detections.json --facts-only
```

One photo can answer spatial questions, but movement needs two observations.
The existing webcam last-seen file stores one object per category rather than
a complete simultaneous scene, so it cannot be used as a photo scene for chat.

The first answer on a large scene can take longer while the model processes the
scene. The default LLM timeout is 180 seconds; adjust it with `--llm-timeout`.
The chat requests a 32K context window, so memory use rises with scene size.
Very large scenes are rejected with a suggestion to use a focused `--classes`
list rather than silently dropping facts from the prompt.

Direct positive “what/which items are next to [one item]?” list questions use the
measured neighbor lookup directly, preserving the complete list without an LLM
omission. Broader questions, ambiguous references, and follow-ups go to Ollama.
