"""Extract pose features for every URFD clip (cached to CSV).

    python -m scripts.extract_pose_features [--camera 0] [--stride 2]

Uses MediaPipe in **IMAGE** mode on purpose. VIDEO mode keeps an internal tracker
between calls; reusing one instance across clips carried the previous clip's state
into the next one and produced frozen, anatomically impossible skeletons. IMAGE
mode is stateless, so that class of bug cannot happen. The cost is a full detector
run per frame instead of a cheap tracking update, which is why frames are
subsampled (stride 2 -> 15 fps) — a fall lasts ~0.5-1 s, so 7-15 samples of the
transition is plenty.

Writes ``data/interim/pose/<clip>.csv`` plus ``data/interim/pose/meta.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.video import iter_rgb_frames, RAW_DIR  # noqa: E402
from src.features.pose import PoseExtractor, frame_features  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "interim" / "pose"
SOURCE_FPS = 30.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted((RAW_DIR / f"cam{args.camera}").glob("*.mp4"))
    effective_fps = SOURCE_FPS / args.stride
    print(f"{len(videos)} clips, camera {args.camera}, {effective_fps:.0f} fps")

    extractor = PoseExtractor(upscale=args.upscale, mode="image")
    started = time.time()
    done = 0

    for path in videos:
        out_path = OUT_DIR / f"{path.stem}.csv"
        if out_path.exists() and not args.overwrite:
            continue

        rows = []
        for idx, frame in enumerate(iter_rgb_frames(path, step=args.stride)):
            pose = extractor.process(frame, int(idx * args.stride / SOURCE_FPS * 1000))
            row = {"frame": idx * args.stride}
            row.update(frame_features(pose.landmarks, pose.visibility))
            if pose.detected:
                row["conf"] = pose.score
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        done += 1
        print(
            f"  {done}/{len(videos)}  {path.stem:14s} "
            f"det={int(df['detected'].sum())}/{len(df)}  ({time.time() - started:.0f}s)"
        )

    extractor.close()
    (OUT_DIR / "meta.json").write_text(
        json.dumps(
            {
                "source_fps": SOURCE_FPS,
                "stride": args.stride,
                "fps": effective_fps,
                "upscale": args.upscale,
                "pose_mode": "image",
                "model": "mediapipe pose_landmarker_lite",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"done in {time.time() - started:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
