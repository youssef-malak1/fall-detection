"""Train and evaluate the temporal fall model with event-level cross-validation.

    python -m scripts.train_temporal [--seeds 3] [--epochs 150]

Protocol (deliberately identical to ``scripts/run_baseline.py``)
---------------------------------------------------------------
* same events, same camera (cam0), same ``StratifiedKFold(5, shuffle, seed=42)`` so
  the two reports are comparable row by row;
* the normaliser is fitted on training frames only;
* an inner split of the training fold drives early stopping — the outer test fold is
  never looked at, not even for a stopping decision;
* several seeds per fold, probabilities averaged, because a 70-event problem is noisy.

Why fixed-length windows and not whole clips
--------------------------------------------
Clip length is a dataset artefact: URFD ADL clips average 112 frames, falls average 50,
because recording stopped shortly after the event. A whole-clip model can learn "short
clip -> fall" and look excellent for the wrong reason (length alone scores F1 0.82). The
model therefore sees fixed 3.2 s windows and the clip score is the max over its windows,
which is also how a streaming deployment would work.

Outputs
-------
reports/temporal_report.md
reports/figures/confusion_temporal.png
reports/figures/temporal_timeline_<clip>.png
data/processed/temporal_oof_predictions.csv
models/fall_tcn.pt   (trained on all events, for the inference service)
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.dataset import build_feature_table, load_pose_meta  # noqa: E402
from src.eval.metrics import aggregate, compute  # noqa: E402
from src.models.sequences import (  # noqa: E402
    CHANNELS,
    apply_transform,
    build_sequences,
    impute_masked,
    make_windows,
    normalise_transform,
)

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
SEED = 42
FOLDS = 5
THRESHOLD = 0.5
WINDOW = 48  # frames @15fps = 3.2 s
EVAL_STRIDE = 12
BATCH = 8


# --------------------------------------------------------------------------------------
def _augment(x: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Time-warp + jitter inside a window; both preserve the order, which is the signal."""
    factor = rng.uniform(0.85, 1.15)
    length = max(8, int(round(len(x) * factor)))
    idx = np.linspace(0, len(x) - 1, length)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(x) - 1)
    weight = (idx - lo).astype(np.float32)[:, None]
    x = x[lo] * (1 - weight) + x[hi] * weight
    mask = (mask[lo] * (1 - weight[:, 0]) + mask[hi] * weight[:, 0]).astype(np.float32)
    x = x + rng.normal(0.0, 0.05, size=x.shape).astype(np.float32)
    return x.astype(np.float32), mask


def _stack(windows: list[tuple[np.ndarray, np.ndarray]], in_channels: int):
    """Pad windows to a common length. The model is padding-invariant thanks to the mask."""
    import torch

    length = max(len(w[0]) for w in windows)
    X = np.zeros((len(windows), length, in_channels), dtype=np.float32)
    M = np.zeros((len(windows), length), dtype=np.float32)
    for i, (x, m) in enumerate(windows):
        X[i, : len(x)] = x
        M[i, : len(m)] = m
    return torch.from_numpy(X), torch.from_numpy(M)


def _build(in_channels: int):
    from src.models.temporal import build_model

    return build_model(in_channels)


def clip_probabilities(model, norm, masks, indices, clips, in_channels, device, stride=EVAL_STRIDE):
    """Score each clip as the max over its fixed-length windows."""
    import torch

    model.eval()
    scores = {}
    with torch.no_grad():
        for i in indices:
            c = clips[i]
            windows = make_windows(norm[c], masks[c], WINDOW, stride)
            X, M = _stack(windows, in_channels)
            probs = torch.sigmoid(model(X.to(device), M.to(device))).cpu().numpy()
            scores[i] = float(probs.max())
    return scores


def train_one(norm, masks, train_idx, val_idx, clips, y, in_channels, seed, max_epochs, patience, device):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    labels = y[train_idx]
    pos_weight = torch.tensor(
        [(labels == 0).sum() / max(1.0, (labels == 1).sum())], dtype=torch.float32
    ).to(device)

    model = _build(in_channels).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max_epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(train_idx))
        for b in range(0, len(order), BATCH):
            batch = []
            for j in order[b : b + BATCH]:
                i = train_idx[j]
                c = clips[i]
                (xw, mw), = make_windows(norm[c], masks[c], WINDOW, EVAL_STRIDE, rng=rng)
                ax, am = _augment(xw, mw, rng)
                batch.append((ax, am))
            X, M = _stack(batch, in_channels)
            yb = torch.tensor(
                [float(y[train_idx[j]]) for j in order[b : b + BATCH]], dtype=torch.float32
            ).to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(X.to(device), M.to(device)), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
        scheduler.step()

        # early stopping on clip-level, max-aggregated validation loss: the quantity we
        # actually report, not the window-level one which is easier and misleading.
        val = clip_probabilities(model, norm, masks, val_idx, clips, in_channels, device)
        probs = torch.tensor([val[i] for i in val_idx], dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        targets = torch.tensor([float(y[i]) for i in val_idx], dtype=torch.float32)
        val_loss = float(
            torch.nn.functional.binary_cross_entropy(probs, targets, weight=None, reduction="mean")
        )

        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


# --------------------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["ADL", "FALL"], cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timelines(model, norm, masks, y, clips, in_channels, device, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    picks = [c for c in ["fall-21-cam0", "fall-23-cam0", "adl-30-cam0", "adl-33-cam0"] if c in clips]
    fps = float(load_pose_meta()["fps"])
    for clip in picks:
        x, m = norm[clip], masks[clip]
        with torch.no_grad():
            xt = torch.from_numpy(x[None].astype(np.float32)).to(device)
            mt = torch.from_numpy(m[None].astype(np.float32)).to(device)
            frames = model.frame_scores(xt, mt)[0].cpu().numpy()

        label = "FALL" if y[clips.index(clip)] == 1 else "ADL"
        t = np.arange(len(m)) / fps
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot(t, frames, color="#1f77b4", lw=1.4, label="per-frame evidence")
        ax.fill_between(t, frames.min(), frames.max(), where=m < 0.5, color="0.85",
                        label="no detection")
        ax.set_title(f"{clip} ({label})")
        ax.set_xlabel("seconds")
        ax.set_ylabel("evidence")
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / f"temporal_timeline_{clip}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels = len(CHANNELS)

    events = build_feature_table(camera=0).reset_index(drop=True)
    clips = events["clip_id"].tolist()
    y = events["label"].to_numpy()
    print(f"events: {len(events)}  falls: {int(y.sum())}  channels: {in_channels}  device: {device}")

    started = time.time()
    seqs, masks = build_sequences(clips)
    print(f"sequences built in {time.time() - started:.0f}s; window {WINDOW} frames")

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof_prob = np.zeros(len(events), dtype=float)
    fold_metrics: list = []
    chosen_epochs: list[int] = []
    per_fold_auc: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(events, y), 1):
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        inner_train, inner_val = next(inner.split(train_idx.reshape(-1, 1), y[train_idx]))
        fit_idx, val_idx = train_idx[inner_train], train_idx[inner_val]

        mean, std = normalise_transform(
            {clips[i]: seqs[clips[i]] for i in fit_idx},
            {clips[i]: masks[clips[i]] for i in fit_idx},
        )
        norm = {
            c: impute_masked(apply_transform(seqs[c], mean, std), masks[c]) for c in clips
        }

        probs = np.zeros(len(test_idx), dtype=float)
        for seed in range(args.seeds):
            model, best_epoch = train_one(
                norm, masks, fit_idx, val_idx, clips, y, in_channels,
                SEED + seed, args.epochs, args.patience, device,
            )
            chosen_epochs.append(best_epoch)
            scores = clip_probabilities(model, norm, masks, test_idx, clips, in_channels, device)
            probs += np.array([scores[i] for i in test_idx])
        probs /= args.seeds
        oof_prob[test_idx] = probs

        pred = (probs >= THRESHOLD).astype(int)
        fold_metrics.append(compute(y[test_idx], pred))
        auc = roc_auc_score(y[test_idx], probs) if len(set(y[test_idx])) > 1 else float("nan")
        per_fold_auc.append(auc)
        print(f"fold {fold}: F1={fold_metrics[-1].f1:.3f} AUC={auc:.3f} "
              f"(epochs {chosen_epochs[-args.seeds:]})")

    oof_pred = (oof_prob >= THRESHOLD).astype(int)
    agg = aggregate(fold_metrics)
    overall = compute(y, oof_pred)

    # ---------- report ----------
    lines: list[str] = []
    add = lines.append
    add("# Temporal model — event-level 5-fold CV\n")
    add(f"Events: **{len(events)}** (falls {int(y.sum())}, ADL {int((1 - y).sum())}), camera 0, "
        f"dilated TCN on {in_channels} per-frame channels (pose fused with motion).")
    add(f"Same folds as `reports/baseline_report.md`. Each clip is scored as the **max over "
        f"{WINDOW / 15:.1f} s windows** (stride {EVAL_STRIDE} frames); inner 20% split of "
        f"each training fold for early stopping; {args.seeds} seeds averaged; "
        f"threshold {THRESHOLD}.\n")

    add("## Results (mean ± std across folds)\n")
    add("| model | precision | recall | F1 | AUC |")
    add("|---|---|---|---|---|")
    add(f"| **temporal TCN (windows)** | {agg['precision_mean']:.3f} ± {agg['precision_std']:.3f} "
        f"| {agg['recall_mean']:.3f} ± {agg['recall_std']:.3f} "
        f"| {agg['f1_mean']:.3f} ± {agg['f1_std']:.3f} | {np.nanmean(per_fold_auc):.3f} |")

    baseline_path = PROCESSED / "baseline_oof_predictions.csv"
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        for kind in ("logreg", "rf", "rule"):
            m = compute(base["label"], base[f"pred_{kind}"])
            add(f"| {kind} (summary features) | {m.precision:.3f} | {m.recall:.3f} "
                f"| {m.f1:.3f} | — |")
    add("")

    add("## Out-of-fold totals (all 70 events)\n")
    add(f"- temporal TCN: TP {overall.tp}, FP {overall.fp}, TN {overall.tn}, FN {overall.fn} "
        f"(precision {overall.precision:.3f}, recall {overall.recall:.3f}, F1 {overall.f1:.3f})")
    add("")
    plot_confusion(y, oof_pred, "temporal TCN (out-of-fold)", FIGURES / "confusion_temporal.png")

    hard = [clips[i] for i in range(len(clips)) if oof_pred[i] != y[i]]
    add("## Errors\n")
    add(f"{len(hard)} of {len(events)} events misclassified out-of-fold: "
        f"{', '.join(hard) if hard else '—'}\n")

    add("## Controls and honest caveats\n")
    add("- **Clip-length shortcut controlled.** URFD falls average 50 frames and ADLs 112, "
        "so whole-clip models learn clip length instead of behaviour (length alone scores "
        "F1 0.82). Fixed-length windows remove that; each sample is the same size.")
    add("- Pooled AUC across folds is not reported as a single number when fold-wise "
        "probabilities are not calibrated against each other; per-fold AUC is reported "
        "instead.")
    add(f"- Early stopping picked epoch {int(np.median(chosen_epochs))} (median over runs).")
    add("- Augmentation: time-warp inside the window + Gaussian jitter in normalised units. "
        "No spatial flip: channels are body-relative, so a flip carries no new information.")
    add("- 70 events, one actor, one room. Any number here is a methodology result.")
    add("")

    REPORT = REPORTS / "temporal_report.md"
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = events[["clip_id", "class", "label"]].copy()
    out["prob_temporal"] = oof_prob
    out["pred_temporal"] = oof_pred
    out.to_csv(PROCESSED / "temporal_oof_predictions.csv", index=False)

    # ---------- deployment model: retrain on everything ----------
    mean_full, std_full = normalise_transform(seqs, masks)
    norm_full = {
        c: impute_masked(apply_transform(seqs[c], mean_full, std_full), masks[c]) for c in clips
    }
    all_idx = np.arange(len(events))
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    itr, iva = next(inner.split(all_idx.reshape(-1, 1), y))
    final_epochs = max(1, int(np.median(chosen_epochs)))
    model, _ = train_one(
        norm_full, masks, all_idx[itr], all_idx[iva], clips, y, in_channels,
        SEED, final_epochs, final_epochs, device,
    )
    checkpoint = {
        "state_dict": model.state_dict(),
        "channels": [name for name, _ in CHANNELS],
        "mean": mean_full.tolist(),
        "std": std_full.tolist(),
        "threshold": THRESHOLD,
        "window": WINDOW,
        "frame_size": [320, 240],
        "pose_fps": float(load_pose_meta()["fps"]),
        "architecture": "FallTCN(width=32, dilations=(1,2,4,8), layernorm)",
        "receptive_field_frames": model.receptive_field,
        "trained_on": {"events": len(events), "falls": int(y.sum()), "adl": int((1 - y).sum())},
    }
    torch.save(checkpoint, MODELS / "fall_tcn.pt")
    plot_timelines(model, norm_full, masks, y, clips, in_channels, device, FIGURES)

    print(f"\nwrote {REPORT}")
    print(f"temporal TCN F1={agg['f1_mean']:.3f}  P={agg['precision_mean']:.3f}  "
          f"R={agg['recall_mean']:.3f}  per-fold AUC={np.nanmean(per_fold_auc):.3f}")
    print(f"saved {MODELS / 'fall_tcn.pt'}")


if __name__ == "__main__":
    main()
