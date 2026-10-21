"""Extract motion features for every URFD clip.

    python -m scripts.extract_motion_features

Outputs
-------
data/interim/motion/<clip>.csv     per-frame features (cached, not in git)
data/processed/motion_summary.csv   one row per clip
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.motion import extract_frame_features, summarise  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw" / "urfd"
INTERIM = PROJECT_ROOT / "data" / "interim" / "motion"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def parse_name(path: Path) -> tuple[str, str, int]:
    parts = path.stem.split("-")
    return parts[0], f"{parts[0]}-{parts[1]}", int(parts[2].replace("cam", ""))


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    videos = sorted(RAW.rglob("*.mp4"))
    rows = []
    started = time.time()

    for i, path in enumerate(videos, 1):
        cls, sequence, camera = parse_name(path)
        cache = INTERIM / f"{path.stem}.csv"

        if cache.exists():
            df = pd.read_csv(cache)
        else:
            df = extract_frame_features(path)
            df.to_csv(cache, index=False)

        row = {
            "file": str(path.relative_to(RAW)).replace("\\", "/"),
            "class": cls,
            "sequence": sequence,
            "camera": camera,
            "frames": len(df),
        }
        row.update(summarise(df))
        rows.append(row)

        if i % 10 == 0 or i == len(videos):
            print(f"{i}/{len(videos)}  ({time.time() - started:.1f}s)")

    summary = pd.DataFrame(rows)
    out = PROCESSED / "motion_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nwrote {out}  shape={summary.shape}")


if __name__ == "__main__":
    main()
