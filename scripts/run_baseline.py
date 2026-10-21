"""Train/evaluate the baseline models with event-level cross-validation.

    python -m scripts.run_baseline

Models compared
---------------
1. ``rule``  - the interpretable state machine in ``src.models.rule_baseline``
2. ``logreg`` - logistic regression on summary features
3. ``rf``     - random forest on summary features

All splits are **event-level** and stratified. The same event never appears in both
train and test (the cam0/cam1 duplicate problem is avoided by using cam0 only).

Outputs
-------
reports/baseline_report.md
data/processed/baseline_oof_predictions.csv
reports/figures/confusion_<model>.png
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.dataset import (  # noqa: E402
    FEATURES,
    build_feature_table,
    load_pose,
    load_pose_meta,
)
from src.eval.metrics import aggregate, compute  # noqa: E402
from src.models.rule_baseline import RuleParams, detect_fall  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"
SEED = 42
FOLDS = 5

GRID = {
    "max_transition_s": [0.6, 0.9, 1.2, 1.5],
    "min_hip_drop": [0.05, 0.10, 0.15],
    "horizontal_angle": [60.0, 70.0, 80.0],
    "min_lying_s": [0.5, 1.0, 2.0],
}


def rule_predictions(pose_cache: dict[str, pd.DataFrame], params: RuleParams, clips) -> np.ndarray:
    return np.array([int(detect_fall(pose_cache[c], params).is_fall) for c in clips])


def tune_rule(pose_cache, train_clips, y_train, fps: float) -> tuple[RuleParams, float]:
    best_params, best_score = RuleParams(fps=fps), -1.0
    for combo in itertools.product(*GRID.values()):
        params = RuleParams(fps=fps, **dict(zip(GRID.keys(), combo)))
        pred = rule_predictions(pose_cache, params, train_clips)
        m = compute(y_train, pred)
        # F1 first, precision as tie-breaker (fewer false alarms)
        score = m.f1 + 0.001 * m.precision
        if score > best_score:
            best_score, best_params = score, params
    return best_params, best_score


def ml_model(kind: str):
    if kind == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
        ),
    )


def plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["ADL", "FALL"], cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    events = build_feature_table(camera=0).reset_index(drop=True)
    y = events["label"].to_numpy()
    clips = events["clip_id"].tolist()
    print(f"events: {len(events)}  falls: {int(y.sum())}  adl: {int((1 - y).sum())}")

    pose_cache = {c: load_pose(c) for c in clips}
    pose_fps = float(load_pose_meta()["fps"])
    print(f"pose fps: {pose_fps:g}")

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = {"rule": np.zeros(len(events), dtype=int), "logreg": np.zeros(len(events), dtype=int), "rf": np.zeros(len(events), dtype=int)}
    fold_metrics: dict[str, list] = {"rule": [], "logreg": [], "rf": []}
    chosen_params: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(events, y), 1):
        train_clips = [clips[i] for i in train_idx]
        test_clips = [clips[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- rule: tune on train, evaluate on test
        params, train_score = tune_rule(pose_cache, train_clips, y_train, pose_fps)
        pred_rule = rule_predictions(pose_cache, params, test_clips)
        oof["rule"][test_idx] = pred_rule
        fold_metrics["rule"].append(compute(y_test, pred_rule))
        chosen_params.append(
            {
                "fold": fold,
                "max_transition_s": params.max_transition_s,
                "min_hip_drop": params.min_hip_drop,
                "horizontal_angle": params.horizontal_angle,
                "min_lying_s": params.min_lying_s,
                "train_f1": round(train_score, 3),
            }
        )

        # --- ML models
        X = events[FEATURES].to_numpy(dtype=float)
        for kind in ("logreg", "rf"):
            model = ml_model(kind)
            model.fit(X[train_idx], y_train)
            pred = model.predict(X[test_idx]).astype(int)
            oof[kind][test_idx] = pred
            fold_metrics[kind].append(compute(y_test, pred))

        print(f"fold {fold}: rule_params={params.max_transition_s}s/{params.min_hip_drop}/"
              f"{params.horizontal_angle}/{params.min_lying_s}")

    # ---------- report ----------
    lines: list[str] = []
    add = lines.append
    add("# Baseline evaluation — event level, 5-fold CV\n")
    add(f"Events: **{len(events)}** (falls {int(y.sum())}, ADL {int((1 - y).sum())}), camera 0 only.")
    add("Splits are stratified by event; the rule thresholds are tuned **inside each "
        "training fold** and evaluated on the untouched test fold.\n")

    add("## Results (mean ± std across folds)\n")
    add("| model | precision | recall | F1 | accuracy | TP | FP | TN | FN |")
    add("|---|---|---|---|---|---|---|---|---|")
    summary_rows = {}
    for kind in ("rule", "logreg", "rf"):
        agg = aggregate(fold_metrics[kind])
        summary_rows[kind] = agg
        add(
            f"| {kind} | {agg['precision_mean']:.3f} ± {agg['precision_std']:.3f} "
            f"| {agg['recall_mean']:.3f} ± {agg['recall_std']:.3f} "
            f"| {agg['f1_mean']:.3f} ± {agg['f1_std']:.3f} "
            f"| {agg['accuracy_mean']:.3f} | {agg['tp']} | {agg['fp']} | {agg['tn']} | {agg['fn']} |"
        )
    add("")

    # pooled out-of-fold confusion matrices
    add("## Out-of-fold confusion matrices (all events)\n")
    for kind in ("rule", "logreg", "rf"):
        m = compute(y, oof[kind])
        add(f"**{kind}** — TP {m.tp}, FP {m.fp}, TN {m.tn}, FN {m.fn} "
            f"(precision {m.precision:.3f}, recall {m.recall:.3f}, F1 {m.f1:.3f})\n")
        plot_confusion(y, oof[kind], f"{kind} (out-of-fold)", FIGURES / f"confusion_{kind}.png")

    # false alarm rate
    adl_seconds = float(events.loc[events.label == 0, "frames"].sum() / 30)
    hours = adl_seconds / 3600
    add("## False alarms — observed, not extrapolated\n")
    add(f"ADL footage in the evaluation set: **{adl_seconds:.0f} s = {hours:.3f} h**.")
    add("")
    for kind in ("rule", "logreg", "rf"):
        fp = summary_rows[kind]["fp"]
        add(f"- {kind}: **{fp} false alarms** in {adl_seconds:.0f} s of ADL "
            f"({fp / max(hours, 1e-9):.0f} per hour of ADL at this operating point)")
    add("")
    add("**Do not quote a per-day false-alarm rate from this.** Five minutes of ADL "
        "footage cannot estimate a deployment false-alarm rate: one more or one fewer "
        "false alarm would move the number by hundreds. The counts above are what was "
        "measured; a real FAR needs continuous recordings of hours, which this dataset "
        "does not contain.")
    add("")

    add("## Tuned rule thresholds per fold\n")
    add("| " + " | ".join(chosen_params[0].keys()) + " |")
    add("|" + "---|" * len(chosen_params[0]))
    for row in chosen_params:
        add("| " + " | ".join(str(v) for v in row.values()) + " |")
    add("")

    add("## What carries the result (see reports/ablation_report.md)\n")
    add("| features | logreg F1 | rf F1 |")
    add("|---|---|---|")
    add("| motion only | 0.858 | 0.791 |")
    add("| pose only | 0.861 | 0.833 |")
    add(f"| motion + pose | {summary_rows['logreg']['f1_mean']:.3f} | "
        f"{summary_rows['rf']['f1_mean']:.3f} |")
    add("")
    add("Motion geometry and pose geometry are individually decent but together "
        "clearly better, i.e. the two views are complementary rather than redundant.\n")

    add("## Known limitations (read before quoting these numbers)\n")
    add(f"- {len(events)} events, one camera, one actor, one room. This is a "
        "methodology baseline, not a performance claim about real deployments.")
    add("- MediaPipe is used as a frozen pose extractor, so the pose model has never "
        "seen this dataset (no leakage), but it *was* trained on much larger data.")
    add("- Falls that happen **towards the camera** (e.g. fall-21, fall-23) are "
        "foreshortened: the torso never becomes horizontal in the image, so both the "
        "rule and the features lose the signal. A second camera or a 3D lift is "
        "required to catch these.")
    add("- Going-to-the-floor ADLs (adl-30 is the documented one, adl-33 behaves the "
        "same) are geometrically identical to a fall at the moment of descent. Only "
        "the *context* distinguishes them: a fall is not preceded by a controlled "
        "crouch and is not followed by the person standing back up. That is exactly "
        "the job of the temporal model in the next phase.")
    add("- Per-clip evaluation only: continuous streaming and multi-person scenes are "
        "not covered here.")
    add("")

    REPORT = REPORTS / "baseline_report.md"
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # save predictions for error analysis
    out = events[["clip_id", "class", "label", "frames"]].copy()
    for kind in ("rule", "logreg", "rf"):
        out[f"pred_{kind}"] = oof[kind]
    out.to_csv(PROCESSED / "baseline_oof_predictions.csv", index=False)

    print(f"\nwrote {REPORT}")
    for kind in ("rule", "logreg", "rf"):
        agg = summary_rows[kind]
        print(f"{kind:7s} F1={agg['f1_mean']:.3f}  P={agg['precision_mean']:.3f}  "
              f"R={agg['recall_mean']:.3f}")


if __name__ == "__main__":
    main()
