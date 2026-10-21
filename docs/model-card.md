# Model card — ElderLife fall detection (v1.0.0)

## What it is

A vision-only fall detector. It watches a person through a fixed camera and answers one
question: **does this short clip contain a fall?**

| | |
|---|---|
| Version | 1.0.0 |
| Task | binary video classification (fall / no fall) |
| Modality | RGB video, single camera, single person |
| Architecture | frozen MediaPipe Pose Landmarker → 19 per-frame features (pose + OpenCV motion) → dilated temporal CNN (`FallTCN`, 32 channels, dilations 1-2-4-8) → clip score = max over 3.2 s windows |
| Parameters | 24,641 (deliberately small; see "Why so small") |
| Checkpoint | `models/fall_tcn.pt` |
| Decision threshold | 0.5 |
| Receptive field | 61 frames ≈ 4.1 s at 15 fps |
| Owner | ElderLife AI — Computer Vision & Fall Detection (Part 3) |

## Intended use

**In scope**

- A fixed camera in a room (elderly home, care facility), one person in frame.
- Raising an alert that a human then verifies. The model is a *screening* component, not
  a medical device and not a replacement for staff.
- Post-hoc analysis of recorded footage (it runs at roughly 1.5–3× realtime on a laptop CPU).

**Out of scope — do not use it for**

- Medical diagnosis, triage, or as the sole basis for a clinical decision.
- Multi-person scenes (the pose model is configured for one person; the largest motion
  blob wins).
- Crowded rooms, handheld/moving cameras, night-vision-only footage.
- Claiming a real-world false-alarm rate: the evaluation data is 8 minutes long.

## Training data

**UR Fall Detection Dataset (URFD)**, University of Rzeszow.

- 70 events: 30 falls, 40 activities of daily living (ADL).
- Camera 0 (RGB) only; camera 1 and the depth sensor are deliberately unused so that the
  split problem (two files per fall) cannot leak.
- 8.3 minutes total, one actor, one room, one camera height.
- License: **CC BY-NC-SA 4.0** — academic, non-commercial use only. This licence
  propagates to the trained weights.
- Citation: Kwolek, B., Kepski, M. (2014). *Human fall detection on embedded platform
  using depth maps and wireless accelerometer.* Computer Methods and Programs in
  Biomedicine, 117(3), 489–501.

## Evaluation

Event-level 5-fold stratified CV, threshold tuned inside each training fold, normalisation
fitted on training folds only, early stopping on an inner split. Numbers are **held-out**.

| model | precision | recall | F1 | per-fold AUC |
|---|---|---|---|---|
| **temporal TCN (deployed)** | 0.943 ± 0.070 | 0.967 ± 0.067 | **0.951 ± 0.040** | 0.992 |
| logistic regression (summary features) | 0.933 | 0.933 | 0.933 | — |
| random forest (summary features) | 0.862 | 0.833 | 0.847 | — |
| rule state machine | 0.846 | 0.733 | 0.786 | — |

Out-of-fold confusion: TP 29, FP 2, TN 38, FN 1.

### What the errors are

3 of 70 events are wrong, and all three are explainable:

- `fall-20` — the person falls from a chair and ends up kneeling/slumped, so the torso
  never looks horizontal. Missed.
- `adl-10` — a benign activity whose window crosses the threshold.
- `adl-33` — the person deliberately goes down to the floor. Geometrically identical to
  a fall at the moment of descent; only the surrounding context distinguishes them.

Two earlier failure modes were fixed rather than hidden: falls *towards* the camera
(`fall-21`, `fall-23`) used to be missed by both the baseline and the rule, and the
temporal model now catches them because the descent is visible as a sequence even when
the final posture is foreshortened.

### A control that changed the design

Clip length is a dataset artefact: URFD falls average 50 frames, ADLs 112, because
recording stopped soon after the event. A whole-clip classifier can exploit that —
**clip length alone scores F1 0.82**. The deployed model therefore scores fixed 3.2 s
windows, which removes the shortcut. The whole-clip variant scored 0.966; the honest
number is 0.951.

## Limitations

1. **One actor, one room, 8 minutes.** There is no demographic or environmental diversity.
   Expect performance to drop before it is measured on real residents.
2. **Falls towards the camera** are foreshortened. A single frontal view loses the
   horizontal torso. Mitigation would be a second camera or 3D lifting — not implemented.
3. **Deliberate going-to-the-floor activities** (lying down to exercise, picking something
   up from the floor at length) are the main false-alarm source.
4. **One person only.** A second person entering the frame is not tracked.
5. **Static camera assumed.** The motion channel estimates the background as the median
   of the clip; a moving camera invalidates it. The pose channel still works.
6. **No temporal context beyond ~4 s.** "Stays on the floor afterwards" is only partly
   captured; a longer-horizon model is the obvious next step.
7. **Latency** is seconds per clip on CPU, because MediaPipe runs per frame. A live
   deployment should score sliding windows as they arrive (the alerting layer already
   supports this), not re-analyse the whole buffer.

## Ethical and privacy notes

- Video of residents is sensitive personal data. The service returns **scores and
  timing, not images**, and does not persist uploads (temporary file deleted per request).
- A missed fall is a safety risk; a false alarm erodes trust in the system. The threshold
  (0.5) is a starting point, not a tuned clinical operating point. Raising it reduces
  false alarms at the cost of recall — that trade-off belongs to the care provider.
- The training data is academic and non-commercial (CC BY-NC-SA 4.0). Commercial use
  requires a different dataset.

## Why the model is so small

70 events. A model with more parameters than examples memorises the actor. The TCN has
24,641 parameters, four dilated blocks, and global pooling — the smallest architecture that
can still represent "upright → fast descent → stays down". Deep video models were
considered and rejected: they cannot be trained honestly on 8 minutes of footage.

## Reproduce

```powershell
python -m scripts.extract_motion_features
python -m scripts.extract_pose_features --camera 0 --upscale 2 --stride 2 --overwrite
python -m scripts.run_baseline          # rule + logreg + random forest
python -m scripts.ablate_features       # which features carry the result
python -m scripts.train_temporal        # the model in this card
python -m scripts.error_analysis        # per-clip errors
python -m scripts.demo_inference        # end-to-end demo
```
