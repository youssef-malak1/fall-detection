"""A small dilated temporal CNN for fall detection on pose+motion sequences.

Why this shape and not an LSTM
------------------------------
With 70 events, the model has to be tiny or it will memorise the actors. A stack of
dilated convolutions with global pooling is the smallest architecture that can still
express "upright, then a fast descent, then stays down":

* dilation grows the receptive field **exponentially** with depth, so four blocks of
  kernel 5 see ``1 + 4*(1+2+4+8) = 61`` frames ≈ 4 s at 15 fps — the scale of a real
  fall — for only a few thousand parameters;
* convolutions are far less data-hungry than recurrence on short sequences, and they
  train faster on CPU;
* global pooling makes the model independent of clip length, so a 1 s clip and a
  30 s clip need no resampling.

The output is a single logit per clip ("does this clip contain a fall"), and the
per-frame activations can be plotted as a timeline, which is what makes the result
explainable to a human reviewer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedPool(nn.Module):
    """Pool over time while ignoring frames where nobody was detected."""

    def __init__(self, last_frames: int = 15) -> None:
        super().__init__()
        self.last_frames = last_frames

    def _tail(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean of the last ``last_frames`` *valid* frames of each clip.

        This has to be driven by the mask, not by array position: clips are padded to
        the longest one in the batch, so ``h[..., -15:]`` would silently average
        padding whenever a clip is shorter than the batch maximum.
        """
        out = torch.zeros(h.shape[0], h.shape[1], device=h.device, dtype=h.dtype)
        for b in range(h.shape[0]):
            valid = torch.nonzero(mask[b] > 0.5, as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            out[b] = h[b][:, valid[-self.last_frames :]].mean(dim=-1)
        return out

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # h: (B, C, T), mask: (B, T)
        m = mask.unsqueeze(1)
        denom = m.sum(dim=-1).clamp(min=1.0)
        mean = (h * m).sum(dim=-1) / denom
        pooled_max = h.masked_fill(m == 0, -1e4).max(dim=-1).values
        tail = self._tail(h, mask)
        return torch.cat([mean, pooled_max, tail], dim=1)


class ResidualDilatedBlock(nn.Module):
    """Dilated conv + per-frame LayerNorm, mask-aware at every step.

    LayerNorm is applied **per frame over the channel axis**, so its statistics never
    mix in other time steps. GroupNorm/BatchNorm would compute statistics across time
    and therefore across the padded region, which makes a clip's activations depend on
    the length of the longest clip in its batch.
    """

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=5, dilation=dilation, padding=2 * dilation
        )
        self.norm = nn.LayerNorm(channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.conv(x) * mask
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.act(h))
        return (x + h) * mask


class FallTCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
        last_frames: int = 15,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(nn.Conv1d(in_channels, width, kernel_size=1), nn.ReLU())
        self.blocks = nn.ModuleList(
            ResidualDilatedBlock(width, d, dropout) for d in dilations
        )
        self.pool = MaskedPool(last_frames=last_frames)
        self.head = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )
        self.receptive_field = 1 + 4 * sum(dilations)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) already normalised. Returns logits (B,)."""
        h = x.transpose(1, 2)  # (B, C, T)
        m = mask.unsqueeze(1)
        h = self.stem(h) * m
        for block in self.blocks:
            # The mask is re-applied inside and after every block: activations of padded
            # frames must never reach a real frame, otherwise a clip's score would depend
            # on which other clips happened to share its batch.
            h = block(h, m)
        return self.head(self.pool(h, mask)).squeeze(-1)

    @torch.no_grad()
    def frame_scores(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-frame contribution to the decision, for timelines and debugging.

        The head consumes a pooled vector, so there is no single per-frame logit. The
        honest projection is the weight the head gives to the *mean* pooling path, which
        turns each frame's features into the share of evidence it contributes.
        """
        h = x.transpose(1, 2)
        m = mask.unsqueeze(1)
        h = self.stem(h) * m
        for block in self.blocks:
            h = block(h, m)
        width = h.shape[1]
        # Collapse the two head layers into one vector over embedding channels, taking
        # only the mean-pooling block of the head's input. Then each frame gets a scalar:
        # how much it pushes the decision, through the mean path, on its own.
        mean_weights = self.head[0].weight[:, :width]  # (width_out, width)
        out_weights = self.head[3].weight  # (1, width_out)
        g = (out_weights @ mean_weights).squeeze(0)  # (width,)
        return torch.einsum("bct,c->bt", h, g)


def build_model(in_channels: int, **kwargs) -> FallTCN:
    return FallTCN(in_channels=in_channels, **kwargs)
