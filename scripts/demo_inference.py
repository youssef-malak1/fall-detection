"""Demo: run the deployed model on URFD clips and produce a visual timeline.

    python -m scripts.demo_inference
    python -m scripts.demo_inference --clips fall-10-cam0 adl-30-cam0

This uses the *deployed* checkpoint, which was retrained on all 70 events. Numbers from
this script are therefore **illustrative, not an evaluation** — the honest numbers are in
``reports/temporal_report.md``, produced with held-out folds.

It also pushes the window scores through the alerting state machine, so the output shows
what the backend would actually receive.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.video import RAW_DIR  # noqa: E402
from src.inference.alerting import FallEventTracker  # noqa: E402
from src.inference.pipeline import FallDetector  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"

DEFAULT_CLIPS = ["fall-01-cam0", "fall-10-cam0", "adl-01-cam0", "adl-30-cam0"]


def plot(analysis, clip: str, out_dir: Path) -> Path:
    times = [w.start_s for w in analysis.windows]
    scores = [w.score for w in analysis.windows]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(times, scores, marker="o", ms=3, lw=1.2, color="#1f77b4")
    ax.axhline(analysis.threshold, color="crimson", ls="--", lw=1,
               label=f"threshold {analysis.threshold:.2f}")
    ax.fill_between(times, 0, scores, where=np.array(scores) >= analysis.threshold,
                    color="crimson", alpha=0.2, label="alert region")
    verdict = "FALL" if analysis.is_fall else "no fall"
    ax.set_title(f"{clip} — {verdict} (score {analysis.score:.3f}, confidence "
                 f"{analysis.model.get('version', '')})")
    ax.set_xlabel("seconds")
    ax.set_ylabel("P(fall in window)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    path = out_dir / f"inference_{clip}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    detector = FallDetector()

    rows = []
    print(f"{'clip':16s} {'verdict':8s} {'score':>6s} {'onset':>6s} {'cover':>6s} "
          f"{'alert':>6s} {'latency':>8s}")
    for clip in args.clips:
        path = RAW_DIR / f"cam{args.camera}" / f"{clip}.mp4"
        if not path.exists():
            print(f"{clip:16s} missing")
            continue
        started = time.perf_counter()
        analysis = detector.analyze_video(path, layout="urfd")
        wall = time.perf_counter() - started
        # A tracker belongs to one camera stream, not to the process: two different clips
        # are two different rooms as far as the alerting layer is concerned.
        tracker = FallEventTracker(
            threshold=detector.threshold, confirm_windows=1, cooldown_s=30
        )
        decision = tracker.update(analysis.onset_s or 0.0, analysis.score)
        plot(analysis, clip, FIGURES)
        print(f"{clip:16s} {'FALL' if analysis.is_fall else 'no fall':8s} "
              f"{analysis.score:6.3f} {analysis.onset_s or 0:6.2f} "
              f"{analysis.pose_coverage:6.2f} {str(decision.alert):>6s} "
              f"{analysis.latency_ms / 1000:7.2f}s")
        rows.append(
            {
                "clip": clip,
                "verdict": "FALL" if analysis.is_fall else "no fall",
                "score": analysis.score,
                "onset_s": analysis.onset_s,
                "pose_coverage": analysis.pose_coverage,
                "alert": decision.alert,
                "latency_s": analysis.latency_ms / 1000,
                "realtime_factor": analysis.duration_s / max(wall, 1e-6),
            }
        )

    detector.close()

    lines = ["# Inference demo\n",
             "Deployed checkpoint retrained on all 70 events, so these numbers illustrate "
             "the plumbing — they are **not** an evaluation. See "
             "`reports/temporal_report.md` for held-out numbers.\n",
             "| clip | verdict | score | onset (s) | pose coverage | alert | latency (s) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['clip']} | {r['verdict']} | {r['score']:.3f} | "
            f"{(r['onset_s'] or 0):.2f} | {r['pose_coverage']:.2f} | {r['alert']} | "
            f"{r['latency_s']:.2f} |"
        )
    (REPORTS / "inference_demo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORTS / 'inference_demo.md'}")


if __name__ == "__main__":
    main()
