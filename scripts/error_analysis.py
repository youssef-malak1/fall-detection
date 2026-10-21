"""Per-clip error analysis for the baseline out-of-fold predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PRED = PROJECT_ROOT / "data" / "processed" / "baseline_oof_predictions.csv"
REPORT = PROJECT_ROOT / "reports" / "error_analysis.md"


def main() -> None:
    d = pd.read_csv(PRED)
    lines = ["# Error analysis (out-of-fold)\n"]
    for model in ("rule", "logreg", "rf"):
        col = f"pred_{model}"
        fp = d[(d.label == 0) & (d[col] == 1)]["clip_id"].tolist()
        fn = d[(d.label == 1) & (d[col] == 0)]["clip_id"].tolist()
        lines.append(f"## {model}\n")
        lines.append(f"- false alarms (ADL flagged as fall, {len(fp)}): {', '.join(fp) or '—'}")
        lines.append(f"- missed falls ({len(fn)}): {', '.join(fn) or '—'}\n")
        print(f"{model}: FP={len(fp)} {fp}")
        print(f"{'':>{len(model)}}  FN={len(fn)} {fn}")

    # which clips are wrong for every model?
    wrong = (
        (d["pred_rule"] != d.label).astype(int)
        + (d["pred_logreg"] != d.label).astype(int)
        + (d["pred_rf"] != d.label).astype(int)
    )
    hard = d.loc[wrong == 3, "clip_id"].tolist()
    lines.append("## Wrong for all three models\n")
    lines.append(f"{', '.join(hard) or '—'}\n")
    print(f"\nwrong for all 3: {hard}")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
