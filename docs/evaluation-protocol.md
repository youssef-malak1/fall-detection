# Evaluation protocol

How this module is measured, and why it is measured this way.

## 1. Unit of evaluation: the **event**, not the frame

The platform raises an alert per incident. So the question the model is scored on is:

> Given a whole clip, did the system correctly say "a fall happened" / "no fall happened"?

Frame-level accuracy is reported only as a secondary diagnostic, because with 60/40
frame imbalance a model can look accurate while never detecting a single fall.

## 2. Splitting: event-level, stratified, 5-fold

Two leakage traps exist in this dataset and both are avoided:

| Trap | Why it leaks | How it is avoided |
|---|---|---|
| Same fall recorded by cam0 and cam1 | Two files = one physical event | Evaluation uses **cam0 only** (70 events) |
| Overlapping windows from one clip | Frames from one fall land in train and test | Splits are made on **clips**, never on frames/windows |

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the 70 events
(30 falls / 40 ADL).

## 3. Tuning is inside the fold

The rule baseline has four thresholds. They are selected by grid search on the
**training folds only**, then applied untouched to the test fold. The reported numbers
are therefore honest out-of-fold estimates, not the best case of a sweep.

## 4. Metrics

| Metric | Why it is here |
|---|---|
| **Recall** (on falls) | Missing a real fall is the dangerous error |
| **Precision** | Every false alarm costs caregiver trust |
| **F1** | Single number for comparing models — the brief asks for it explicitly |
| **Confusion matrix** | Shows *which* mistakes are made, not just how many |
| **AUC (per fold)** | Ranking quality independent of the threshold |
| **Observed false alarms** | Raw count per hour of ADL footage in the evaluation set |

### Why there is no "false alarms per 24 h" number

The evaluation set contains ~298 s of ADL footage — five minutes. A per-day rate computed
from that is a linear extrapolation in which one extra false alarm moves the answer by
hundreds; an earlier draft of this project reported "1160 per 24 h" and the number was
mathematically meaningless. The reports now state the **observed count** and the footage
it was observed on, and say explicitly that a deployment false-alarm rate needs hours of
continuous recording, which this dataset does not contain.

### Why pooled AUC is not reported for the temporal model

Probabilities from different folds come from different models and are not calibrated
against each other, so pooling them can distort AUC. Per-fold AUC is reported instead.

## 5. Mandatory controls (a result is not accepted without them)

1. **Shortcut controls.** URFD falls average 50 frames and ADLs 112, because recording
   stopped soon after the event. Any model that sees the whole clip can learn clip length:
   **length alone scores F1 0.82**. The temporal model therefore scores fixed 3.2 s
   windows. When a new feature or model is added, re-run this control — a jump in F1 that
   coincides with a length-correlated design change is a warning, not a win.
2. **Normalisation fitted on training folds only.** Fitting a scaler on the full dataset
   leaks the test distribution into training.
3. **Early stopping on an inner split of the training fold**, never on the test fold.
4. **Plausibility gates are features, not filters.** Frames with an implausible pose are
   marked (`detected = 0`) and excluded by the mask; they are never silently dropped, so
   coverage stays visible in the report (`pose_detection_ratio`).

## 6. Error analysis is mandatory

Every evaluation run must record:

1. Which events were missed (clip names) and why the rule/model stayed silent.
2. Which ADL clips triggered a false alarm and which normal activity they contain.
3. A statement of what would have to change for those cases to pass.

## 7. Known limitations to repeat in the final report

- 70 events, 8.3 minutes, one room, acted falls by young volunteers.
- No published subject IDs → no subject-wise generalization claim.
- MediaPipe pose is used zero-shot; keypoints degrade for small and lying persons.
- Falls towards the camera are foreshortened and cannot be recovered from one view.
- The model is a **proof of concept**, not a certified medical device.
