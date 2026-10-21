# Temporal model — event-level 5-fold CV

Events: **70** (falls 30, ADL 40), camera 0, dilated TCN on 19 per-frame channels (pose fused with motion).
Same folds as `reports/baseline_report.md`. Each clip is scored as the **max over 3.2 s windows** (stride 12 frames); inner 20% split of each training fold for early stopping; 3 seeds averaged; threshold 0.5.

## Results (mean ± std across folds)

| model | precision | recall | F1 | AUC |
|---|---|---|---|---|
| **temporal TCN (windows)** | 0.943 ± 0.070 | 0.967 ± 0.067 | 0.951 ± 0.040 | 0.992 |
| logreg (summary features) | 0.933 | 0.933 | 0.933 | — |
| rf (summary features) | 0.862 | 0.833 | 0.847 | — |
| rule (summary features) | 0.846 | 0.733 | 0.786 | — |

## Out-of-fold totals (all 70 events)

- temporal TCN: TP 29, FP 2, TN 38, FN 1 (precision 0.935, recall 0.967, F1 0.951)

## Errors

3 of 70 events misclassified out-of-fold: adl-10-cam0, adl-33-cam0, fall-20-cam0

## Controls and honest caveats

- **Clip-length shortcut controlled.** URFD falls average 50 frames and ADLs 112, so whole-clip models learn clip length instead of behaviour (length alone scores F1 0.82). Fixed-length windows remove that; each sample is the same size.
- Pooled AUC across folds is not reported as a single number when fold-wise probabilities are not calibrated against each other; per-fold AUC is reported instead.
- Early stopping picked epoch 25 (median over runs).
- Augmentation: time-warp inside the window + Gaussian jitter in normalised units. No spatial flip: channels are body-relative, so a flip carries no new information.
- 70 events, one actor, one room. Any number here is a methodology result.

