# ElderLife AI — Computer Vision & Fall Detection (Part 3)

Vision-based fall detection and activity recognition module for the **ElderLife AI**
elderly-care platform.

**Scope of this repository: Part 3 only.**
Data selection → preprocessing → baseline → trained model → evaluation → inference API.
Backend / Frontend / Database / Power BI are out of scope; this module only *serves* a
result through a documented interface.

---

## 1. Problem statement

Detect a **fall** of an elderly person from video, and **distinguish it from normal
daily activities (ADL)** — sitting down fast, bending to pick something up, lying down
on a bed/sofa. The output feeds the platform's alerting layer.

Operational definition of a fall used in this project:

> A transition from an upright / standing posture to lying on the **floor**, reached
> quickly, followed by **remaining on the floor** for more than a defined number of
> seconds without getting back up.

Anything that ends in a controlled posture (bed, chair) or is followed by a quick
recovery is **not** a fall.

## 2. Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | Dataset selection + license documentation + data audit | **done** — see `reports/data_audit.md` |
| 2 | Preprocessing pipeline (video → frames → person/pose → sequences) | **done** — `src/data/video.py`, `src/features/` |
| 3 | Rule-based **baseline** fall detector | **done** — `src/models/rule_baseline.py` |
| 4 | Pose-based temporal model | **done** — `src/models/temporal.py`, `reports/temporal_report.md` |
| 5 | Evaluation: Precision / Recall / F1 + False Alarm Rate + error analysis | **done** — `reports/baseline_report.md`, `reports/temporal_report.md`, `reports/error_analysis.md` |
| 6 | Inference service (REST + JSON contract) + Docker | **done** — `src/inference/`, `Dockerfile`, `docs/api-contract.md` |
| 7 | Model card + demo + docs | **done** — `docs/model-card.md`, `scripts/demo_inference.py` |

### How to run

```powershell
# 1. dataset (115 MB, slow host) then audit
powershell -ExecutionPolicy Bypass -File scripts/download_urfd.ps1
python -m src.data.audit_dataset

# 2. features  (motion: ~40 s   |   pose: ~7 min on this machine)
python -m scripts.extract_motion_features
python -m scripts.extract_pose_features --camera 0 --upscale 2 --stride 2

# 3. baselines: rule + logistic regression + random forest, 5-fold event-level CV
python -m scripts.run_baseline
python -m scripts.ablate_features    # which feature group carries the result
python -m scripts.error_analysis     # per-clip false alarms / missed falls

# 4. temporal model (the deployed one) + demo
python -m scripts.train_temporal     # ~4 min on CPU, writes models/fall_tcn.pt
python -m scripts.demo_inference     # end-to-end on a few URFD clips

# 5. REST service
uvicorn src.inference.service:app --port 8100
# or: docker compose up --build

# 6. tests
python -m pytest tests -q            # 44 tests
```

### Phase 0 results (dataset audit)

- **100 video files → 70 unique events** (30 falls, 40 ADL). Falls are recorded from
  two cameras, so the file count double-counts them.
- **8.3 minutes** of footage in total — this is a small dataset, and every design
  decision downstream has to respect that.
- Frames are 640×240 side-by-side composites: **depth on the left, RGB on the right**.
  The usable RGB frame is 320×240.
- Consequence: train a **small temporal head on top of a frozen pretrained pose
  model**; do not attempt to train a video model from scratch.
- All splits must be **event-level**, never file-level.

Full numbers: [`reports/data_audit.md`](reports/data_audit.md) ·
dataset card: [`docs/dataset-card.md`](docs/dataset-card.md)

## 3. Phase 1 results (baseline, event-level 5-fold CV, 70 events)

| model | precision | recall | F1 |
|---|---|---|---|
| rule state machine (tuned per fold) | 0.827 | 0.733 | 0.768 |
| **logistic regression (motion + pose)** | **0.950** | **0.933** | **0.935** |
| random forest (motion + pose) | 0.888 | 0.833 | 0.838 |

Ablation — neither view alone is enough:

| features | logreg F1 | rf F1 |
|---|---|---|
| motion only | 0.858 | 0.791 |
| pose only | 0.861 | 0.833 |
| motion + pose | **0.935** | 0.838 |

What the remaining errors actually are (see
[`reports/error_analysis.md`](reports/error_analysis.md)):

- falls **towards the camera** (`fall-21`, `fall-23`) are foreshortened — the torso
  never looks horizontal, so a single frontal camera physically loses the signal;
- deliberate **going-to-the-floor ADLs** (`adl-30`, `adl-33`) are geometrically
  identical to a fall at the moment of descent — only the context (no controlled
  crouch before, no getting back up after) separates them, which is what the temporal
  model in Phase 2 is for.

Honest caveat: 70 events, one actor, one room, one camera. These numbers are a
methodology baseline, not a deployment claim.

Full report: [`reports/baseline_report.md`](reports/baseline_report.md) ·
pose pipeline pitfalls: [`docs/pose-pipeline.md`](docs/pose-pipeline.md)

## 3b. Phase 2 results — temporal model (the deployed one)

A 24,641-parameter dilated temporal CNN over 19 per-frame channels (pose fused with
motion geometry), scored as the max over 3.2 s windows, same folds as the baseline.

| model | precision | recall | F1 | per-fold AUC |
|---|---|---|---|---|
| **temporal TCN** | 0.943 ± 0.070 | 0.967 ± 0.067 | **0.951 ± 0.040** | 0.992 |
| logistic regression (summary) | 0.933 | 0.933 | 0.933 | — |
| random forest (summary) | 0.862 | 0.833 | 0.847 | — |
| rule state machine | 0.846 | 0.733 | 0.786 | — |

Out-of-fold: TP 29, FP 2, TN 38, FN 1. The three errors are `fall-20` (chair fall ending
kneeling, torso never horizontal), `adl-10`, and `adl-33` (deliberately lies on the floor).

**A result we threw away.** The first version scored the whole clip and reached F1 0.966
with AUC 1.000 in every fold. That was suspicious, and it was: URFD falls average 50
frames while ADLs average 112, so clip length alone scores **F1 0.82**. Fixed-length
windows removed the shortcut; 0.951 is the honest number.

Full report: [`reports/temporal_report.md`](reports/temporal_report.md) ·
model card: [`docs/model-card.md`](docs/model-card.md)

## 3c. Phase 3 — inference service

```powershell
uvicorn src.inference.service:app --port 8100      # or: docker compose up --build
curl -X POST "http://localhost:8100/analyze?layout=urfd" -F "file=@clip.mp4"
```

```json
{
  "schema_version": "1.0",
  "is_fall": true,
  "score": 0.999,
  "threshold": 0.5,
  "confidence": "high",
  "onset_s": 1.6,
  "windows": [{"start_s": 1.6, "end_s": 4.8, "score": 0.999}],
  "pose_coverage": 0.75,
  "latency_ms": 2148.0,
  "model": {"name": "elderlife-fall-tcn", "version": "1.0.0"}
}
```

The backend only needs the JSON: scores, timing and the evidence timeline — never
images. An alerting state machine (`src/inference/alerting.py`) provides the debounce and
cooldown so one fall produces one alert, not one per window. Full contract:
[`docs/api-contract.md`](docs/api-contract.md).

Measured on this machine: **1.5–5 s per clip** (1.5–3× realtime), because MediaPipe runs
per frame. For a live camera, score sliding windows as they arrive rather than
re-uploading a growing buffer.

## 4. Dataset

**UR Fall Detection Dataset (URFD)** — University of Rzeszow.
70 sequences (30 falls, 40 ADL), RGB + depth + accelerometer, 2 cameras.

- Source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
- License: **CC BY-NC-SA 4.0** — non-commercial academic use only.
- Citation: Kwolek, B., Kepski, M. (2014). *Human fall detection on embedded platform
  using depth maps and wireless accelerometer.* Computer Methods and Programs in
  Biomedicine, 117(3), 489–501.

Details, alternatives and comparison: [`docs/data-sources.md`](docs/data-sources.md).

## 5. Repository layout

```
data/
  raw/urfd/          downloaded videos (cam0/, cam1/)   [not in git]
  interim/           extracted frames / keypoints        [not in git]
  processed/         feature tables, index CSVs, splits
  external/          other datasets
notebooks/           exploration
src/
  data/              download + audit + preprocessing
  features/          pose / motion feature extraction
  models/            baseline + temporal model
  eval/              metrics, plots, error analysis
  inference/         service + event state machine
models/              trained weights + model cards
reports/             figures + written reports
configs/             yaml configs
scripts/             one-off helper scripts
docs/                documentation
tests/               unit tests
```

## 6. Setup

Requires Python 3.11 and `ffmpeg`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the dataset (115 MB, slow host):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_urfd.ps1
```

Audit what was downloaded:

```powershell
python -m src.data.audit_dataset
```

## 7. Environment constraints

Development machine: Intel i9-9980HK (8C/16T), **7.7 GB RAM**, NVIDIA Quadro T2000
(**4 GB VRAM**), Intel UHD 630. Heavy training is expected to run on Google Colab;
inference and evaluation must stay light enough for this machine.

## 8. Roadmap

- [x] Phase 0 — toolchain, dataset selection, audit
- [x] Phase 1 — pipeline + rule & ML baselines + evaluation harness + error analysis
- [x] Phase 2 — pose-based temporal model + comparison against the baseline
- [x] Phase 3 — inference API + Docker + demo + model card
