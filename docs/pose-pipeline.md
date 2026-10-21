# Pose pipeline — what we use and the two traps we fell into

`src/features/pose.py` wraps MediaPipe Pose Landmarker (model
`pose_landmarker_lite.task`, shipped in `models/`), and
`scripts/extract_pose_features.py` turns each clip into a per-frame pose CSV in
`data/interim/pose/<clip>.csv`.

## Configuration used for the reported results

| setting | value | why |
|---|---|---|
| running mode | **IMAGE** | stateless — see trap 1 |
| upscale | 2 (640×480) | matches MediaPipe's internal input scale; costs nothing |
| stride | 2 → **15 fps** | a fall lasts 0.5–1 s → 7–15 samples of the transition |
| `min_pose_detection_confidence` | 0.5 | fewer chair-read-as-person hallucinations |
| plausibility gate | `mean_visibility ≥ 0.5` **and** `torso_len ≥ 0.02` | a hallucinated skeleton is worse than no skeleton |

The effective fps is written to `data/interim/pose/meta.json` and read back by
`src/eval/dataset.py` and `scripts/run_baseline.py`, so changing `--stride` cannot
silently desynchronise the temporal thresholds.

## Trap 1 — tracker state leaks between clips (silent data corruption)

MediaPipe `RunningMode.VIDEO` keeps an internal tracker between `detect_for_video`
calls. Reusing one landmarker across 70 different videos carries the previous clip's
state into the next one.

The symptom was *believable-looking data*: for `fall-20` and `fall-21` the extractor
reported `detected = 1` on every frame, but `hip_y` was pinned at 0.83–0.85 and the
torso angle never exceeded 5°, with the ankles above the hips — an anatomically
impossible skeleton sitting still at the bottom of the frame. Feeding that into the
evaluation would have produced a respectable-looking F1 built on fabricated features.

It was caught by drawing the landmarks on frames and comparing them with the numbers
in the CSV (`reports/figures/eda/fall-20-cam0_pose_check.jpg`).

**Fix:** `PoseExtractor(mode="image")`. IMAGE mode has no tracker, so the failure
mode cannot occur. The cost is a full detector pass per frame instead of a cheap
tracking update, which is why frames are subsampled.

## Trap 2 — recreating the model per clip

The first attempted fix was calling `reset()` (recreate the landmarker) before every
clip. Correct, but it turned ~8 min of extraction into ~30 min of CPU churn for no
extra accuracy.

**Fix:** IMAGE mode matches the accuracy at the same cost per frame, and needs no
reset at all.

## Coordinate conventions

Landmarks are normalised to the 320×240 RGB half of the frame (`x`, `y` ∈ [0, 1],
`y` grows downwards). Derived per-frame features:

| column | meaning |
|---|---|
| `torso_angle` | 0° = perfectly upright, 90° = perfectly horizontal (shoulder→hip vector) |
| `hip_y`, `shoulder_y`, `ankle_y`, `head_y` | normalised vertical position of that joint |
| `torso_len`, `pose_h`, `pose_w`, `body_aspect` | body size / orientation in the image |
| `detected` | 1 only if the pose passes the plausibility gate |
| `raw_detected` | what MediaPipe itself returned, kept for debugging |

## How to reproduce

```powershell
python -m scripts.extract_pose_features --camera 0 --upscale 2 --stride 2 --overwrite
```

~7 minutes on the development machine. A clip is only considered valid after
spot-checking landmark overlays against the CSV, which is what caught trap 1.
