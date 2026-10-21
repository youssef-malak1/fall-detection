# Baseline evaluation — event level, 5-fold CV

Events: **70** (falls 30, ADL 40), camera 0 only.
Splits are stratified by event; the rule thresholds are tuned **inside each training fold** and evaluated on the untouched test fold.

## Results (mean ± std across folds)

| model | precision | recall | F1 | accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| rule | 0.827 ± 0.106 | 0.733 ± 0.226 | 0.768 ± 0.184 | 0.829 | 22 | 4 | 36 | 8 |
| logreg | 0.950 ± 0.100 | 0.933 ± 0.082 | 0.935 ± 0.056 | 0.943 | 28 | 2 | 38 | 2 |
| rf | 0.888 ± 0.098 | 0.833 ± 0.183 | 0.838 ± 0.092 | 0.871 | 25 | 4 | 36 | 5 |

## Out-of-fold confusion matrices (all events)

**rule** — TP 22, FP 4, TN 36, FN 8 (precision 0.846, recall 0.733, F1 0.786)

**logreg** — TP 28, FP 2, TN 38, FN 2 (precision 0.933, recall 0.933, F1 0.933)

**rf** — TP 25, FP 4, TN 36, FN 5 (precision 0.862, recall 0.833, F1 0.847)

## False alarms — observed, not extrapolated

ADL footage in the evaluation set: **298 s = 0.083 h**.

- rule: **4 false alarms** in 298 s of ADL (48 per hour of ADL at this operating point)
- logreg: **2 false alarms** in 298 s of ADL (24 per hour of ADL at this operating point)
- rf: **4 false alarms** in 298 s of ADL (48 per hour of ADL at this operating point)

**Do not quote a per-day false-alarm rate from this.** Five minutes of ADL footage cannot estimate a deployment false-alarm rate: one more or one fewer false alarm would move the number by hundreds. The counts above are what was measured; a real FAR needs continuous recordings of hours, which this dataset does not contain.

## Tuned rule thresholds per fold

| fold | max_transition_s | min_hip_drop | horizontal_angle | min_lying_s | train_f1 |
|---|---|---|---|---|---|
| 1 | 1.5 | 0.05 | 60.0 | 0.5 | 0.876 |
| 2 | 1.2 | 0.05 | 60.0 | 0.5 | 0.774 |
| 3 | 1.5 | 0.05 | 60.0 | 0.5 | 0.801 |
| 4 | 1.5 | 0.05 | 60.0 | 0.5 | 0.756 |
| 5 | 1.5 | 0.05 | 60.0 | 0.5 | 0.827 |

## What carries the result (see reports/ablation_report.md)

| features | logreg F1 | rf F1 |
|---|---|---|
| motion only | 0.858 | 0.791 |
| pose only | 0.861 | 0.833 |
| motion + pose | 0.935 | 0.838 |

Motion geometry and pose geometry are individually decent but together clearly better, i.e. the two views are complementary rather than redundant.

## Known limitations (read before quoting these numbers)

- 70 events, one camera, one actor, one room. This is a methodology baseline, not a performance claim about real deployments.
- MediaPipe is used as a frozen pose extractor, so the pose model has never seen this dataset (no leakage), but it *was* trained on much larger data.
- Falls that happen **towards the camera** (e.g. fall-21, fall-23) are foreshortened: the torso never becomes horizontal in the image, so both the rule and the features lose the signal. A second camera or a 3D lift is required to catch these.
- Going-to-the-floor ADLs (adl-30 is the documented one, adl-33 behaves the same) are geometrically identical to a fall at the moment of descent. Only the *context* distinguishes them: a fall is not preceded by a controlled crouch and is not followed by the person standing back up. That is exactly the job of the temporal model in the next phase.
- Per-clip evaluation only: continuous streaming and multi-person scenes are not covered here.

