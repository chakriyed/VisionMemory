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
