"""Feature-group ablation: how much does pose actually add?

    python -m scripts.ablate_features

Runs the same event-level 5-fold CV as the baseline, but trains on different
feature groups so the report can say *which* signal carries the result instead of
only quoting one number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.dataset import build_feature_table  # noqa: E402
from src.eval.metrics import aggregate, compute  # noqa: E402

SEED = 42
FOLDS = 5

MOTION = [
    "detected_ratio",
    "max_horizontal_run",
    "max_lying_run",
    "max_drop_speed",
    "final_centroid_y",
    "final_aspect",
    "min_height_ratio",
]
POSE = [
    "pose_detected_ratio",
    "torso_angle_mean",
    "torso_angle_max",
    "torso_angle_std",
    "hip_y_max",
    "hip_y_range",
    "max_hip_speed",
    "body_aspect_max",
    "conf_mean",
    "longest_horizontal_run",
    "pose_lost_permanently",
]
GROUPS = {"motion": MOTION, "pose": POSE, "motion+pose": MOTION + POSE}


def model(kind: str):
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


def main() -> None:
    events = build_feature_table(camera=0).reset_index(drop=True)
    y = events["label"].to_numpy()
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    rows = []
    for group, cols in GROUPS.items():
        X = events[cols].to_numpy(dtype=float)
        for kind in ("logreg", "rf"):
            fold_metrics = []
            for train_idx, test_idx in skf.split(X, y):
                m = model(kind)
                m.fit(X[train_idx], y[train_idx])
                pred = m.predict(X[test_idx]).astype(int)
                fold_metrics.append(compute(y[test_idx], pred))
            agg = aggregate(fold_metrics)
            rows.append(
                {
                    "features": group,
                    "model": kind,
                    "precision": agg["precision_mean"],
                    "recall": agg["recall_mean"],
                    "f1": agg["f1_mean"],
                    "f1_std": agg["f1_std"],
                }
            )
            print(f"{group:12s} {kind:7s} F1={agg['f1_mean']:.3f}  "
                  f"P={agg['precision_mean']:.3f}  R={agg['recall_mean']:.3f}")

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "reports" / "ablation_report.md"
    lines = ["# Feature-group ablation\n",
             "Same 5-fold event-level CV for every row; only the input features change.\n",
             "| features | model | precision | recall | F1 |",
             "|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(
            f"| {r.features} | {r.model} | {r.precision:.3f} | {r.recall:.3f} | "
            f"{r.f1:.3f} ± {r.f1_std:.3f} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
