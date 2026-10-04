"""Classification losses, class weighting, Mixup and CutMix."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def build_criterion(kind: str = "ce", **kw):
    kind = kind.lower()
    weight = kw.get("weight")
    if weight is not None:
        weight = torch.as_tensor(weight, dtype=torch.float32)
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if kind == "ls":
        return LabelSmoothingCE(
            smoothing=float(kw.get("smoothing", 0.1)),
            weight=weight,
        )
    if kind == "focal":
        return FocalLoss(
            gamma=float(kw.get("gamma", 2.0)),
            alpha=kw.get("alpha", weight),
        )
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted requires a 'weight' tensor")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError("kind must be one of: ce, ls, focal, ce_weighted")


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, weight=None):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = float(smoothing)
        self.register_buffer(
            "weight",
            None if weight is None else torch.as_tensor(weight, dtype=torch.float32),
        )

    def forward(self, logits, target):
        return F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            label_smoothing=self.smoothing,
        )


class FocalLoss(nn.Module):
    """Multi-class focal loss. With gamma=0 and alpha=None this is CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        alpha_tensor = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha_tensor)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.long().view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt).clamp_min(0.0) ** self.gamma) * log_pt
        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            if alpha.ndim == 0:
                loss = loss * alpha
            else:
                loss = loss * alpha.gather(0, target.long())
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Return nine class weights normalized to mean 1, from train counts only."""
    if isinstance(counts, dict):
        values = np.asarray([counts.get(i, 0) for i in range(9)], dtype=np.float64)
    elif isinstance(counts, pd_series_type()):
        values = counts.reindex(range(9), fill_value=0).to_numpy(dtype=np.float64)
    else:
        values = np.asarray(counts, dtype=np.float64).reshape(-1)

    if values.size != 9:
        raise ValueError(f"Expected 9 class counts, got {values.size}")
    if np.any(values <= 0):
        raise ValueError(f"Every class must have a positive train count, got {values}")
    if beta < 0 or beta >= 1:
        raise ValueError("beta must be in [0, 1)")

    if beta == 0:
        weights = 1.0 / values
    else:
        weights = (1.0 - beta) / (-np.expm1(values * math.log(beta)))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def pd_series_type():
    """Avoid importing pandas solely to accept its Series input."""
    try:
        import pandas as pd
        return pd.Series
    except ImportError:
        return ()  # type: ignore[return-value]


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if x.ndim != 4 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected x=(N,C,H,W), y=(N,)")
    mode = mode.lower()
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode must be 'mixup' or 'cutmix'")

    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(x.shape[0], device=x.device)
    y_a, y_b = y, y[permutation]

    if mode == "mixup":
        mixed = lam * x + (1.0 - lam) * x[permutation]
        return mixed, (y_a, y_b, lam)

    mixed = x.clone()
    height, width = x.shape[-2:]
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
    center_x = int(torch.randint(width, (1,), device=x.device).item())
    center_y = int(torch.randint(height, (1,), device=x.device).item())
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)
    mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    actual_lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, (y_a, y_b, actual_lam)


def mixed_loss(criterion, logits, targets):
    if not isinstance(targets, (tuple, list)) or len(targets) != 3:
        return criterion(logits, targets)
    y_a, y_b, lam = targets
    if isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
        # Normalize by the weights of the mixed targets, not by each label
        # group separately; this matches weighted CE's mean reduction.
        weight = criterion.weight.to(device=logits.device, dtype=logits.dtype)
        loss_a = F.cross_entropy(logits, y_a, weight=weight, reduction="none")
        loss_b = F.cross_entropy(logits, y_b, weight=weight, reduction="none")
        normalizer = float(lam) * weight[y_a].sum() + (1.0 - float(lam)) * weight[y_b].sum()
        return (float(lam) * loss_a.sum() + (1.0 - float(lam)) * loss_b.sum()) / normalizer
    return float(lam) * criterion(logits, y_a) + (1.0 - float(lam)) * criterion(logits, y_b)
