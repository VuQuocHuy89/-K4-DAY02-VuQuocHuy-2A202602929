"""Validation-first inference utilities: TTA, calibration, ensembling and BN fusion."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def predict_logits(model, loader, device, view=None):
    """Return ordered filenames, labels and logits for a loader."""
    device = torch.device(device)
    model.eval()
    filenames, labels, logits_out = [], [], []
    with torch.inference_mode():
        for images, target, batch_names in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)
            logits = model(images)
            filenames.extend(list(batch_names))
            labels.append(target.detach().cpu().numpy())
            logits_out.append(logits.float().detach().cpu().numpy())
    if not logits_out:
        raise RuntimeError("Prediction loader produced no batches")
    return filenames, np.concatenate(labels), np.concatenate(logits_out)


def predict_multiview_logits(model, loader, device, make_views, space="prob"):
    """Run every batch through views returned by make_views(batch), then aggregate."""
    device = torch.device(device)
    model.eval()
    filenames, labels, per_view = [], [], None
    with torch.inference_mode():
        for images, target, batch_names in loader:
            images = images.to(device, non_blocking=True)
            batch_views = make_views(images)
            if not isinstance(batch_views, (tuple, list)) or not batch_views:
                raise ValueError("make_views must return a non-empty list of image batches")
            batch_logits = [model(view).float().cpu().numpy() for view in batch_views]
            if per_view is None:
                per_view = [[] for _ in batch_logits]
            if len(batch_logits) != len(per_view):
                raise ValueError("The number of views changed between batches")
            for output, values in zip(per_view, batch_logits):
                output.append(values)
            filenames.extend(list(batch_names))
            labels.append(target.cpu().numpy())
    if per_view is None:
        raise RuntimeError("Prediction loader produced no batches")
    logits_by_view = [np.concatenate(parts) for parts in per_view]
    probs = aggregate_views(logits_by_view, space=space)
    return filenames, np.concatenate(labels), probs


def predict_probabilities(
    model,
    loader,
    device,
    method="I00",
    temperature=1.0,
    aggregation="prob",
    crop_size=224,
    scales=(224, 256),
    return_uncalibrated=False,
):
    """Evaluate one configured inference method and return ordered probabilities.

    Supported methods: I00 (one view), I01 (horizontal flip), I02 (five crops),
    I03 (probability/logit aggregation), I04 (multi-scale), I07 (temperature).
    Use this on validation to choose a method. For final runs it is called on test
    only once after the method and temperature have been frozen.
    """
    method = method.upper()
    supported = {"I00", "I01", "I02", "I03", "I04", "I07"}
    if method not in supported:
        raise ValueError(f"method must be one of {sorted(supported)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    device = torch.device(device)
    model.eval()
    filenames, labels, probabilities, uncalibrated = [], [], [], []

    with torch.inference_mode():
        for images, target, batch_names in loader:
            images = images.to(device, non_blocking=True)
            if method in {"I00", "I07"}:
                logits = model(images).float()
                raw_probs = _softmax_numpy(logits)
            elif method == "I01":
                logits = [model(images).float(), model(view_hflip(images)).float()]
                raw_probs = aggregate_views(logits, space="prob")
            elif method == "I02":
                crops = views_multicrop(images, int(crop_size))
                logits = [model(crop).float() for crop in crops]
                raw_probs = aggregate_views(logits, space="prob")
            elif method == "I03":
                logits = [model(images).float(), model(view_hflip(images)).float()]
                raw_probs = aggregate_views(logits, space=aggregation)
            else:  # I04
                scale_batches = views_multiscale(images, scales)
                logits = [model(view).float() for view in scale_batches]
                raw_probs = aggregate_views(logits, space="prob")

            batch_probs = _temperature_scale_probs(raw_probs, temperature)
            probabilities.append(_as_numpy(batch_probs))
            if return_uncalibrated:
                uncalibrated.append(_as_numpy(raw_probs))
            labels.append(target.cpu().numpy())
            filenames.extend(list(batch_names))

    if not probabilities:
        raise RuntimeError("Prediction loader produced no batches")
    output = (filenames, np.concatenate(labels), np.concatenate(probabilities))
    if return_uncalibrated:
        return (*output, np.concatenate(uncalibrated))
    return output


def view_identity(x):
    return x


def view_hflip(x):
    if x.ndim != 4:
        raise ValueError("Expected image batch with shape (N,C,H,W)")
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """Five deterministic crops: four corners and center."""
    if x.ndim != 4:
        raise ValueError("Expected image batch with shape (N,C,H,W)")
    height, width = x.shape[-2:]
    if crop <= 0 or crop > min(height, width):
        raise ValueError(f"crop must be in [1, {min(height, width)}]")
    top_left = (0, 0)
    positions = [
        top_left,
        (0, width - crop),
        (height - crop, 0),
        (height - crop, width - crop),
        ((height - crop) // 2, (width - crop) // 2),
    ]
    return [x[:, :, top:top + crop, left:left + crop] for top, left in positions]


def views_multiscale(x, sizes):
    """Resize normalized batches to each requested square resolution."""
    if x.ndim != 4:
        raise ValueError("Expected image batch with shape (N,C,H,W)")
    if isinstance(sizes, int):
        sizes = [sizes]
    result = []
    for size in sizes:
        if int(size) <= 0:
            raise ValueError("Every scale must be positive")
        result.append(F.interpolate(
            x, size=(int(size), int(size)), mode="bilinear",
            align_corners=False, antialias=True,
        ))
    return result


def _softmax_numpy(logits):
    values = _as_numpy(logits).astype(np.float64, copy=False)
    values = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=-1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Aggregate view logits by averaging probabilities or logits."""
    if not logits_per_view:
        raise ValueError("At least one view is required")
    arrays = [_as_numpy(item).astype(np.float64, copy=False) for item in logits_per_view]
    shape = arrays[0].shape
    if len(shape) != 2 or any(item.shape != shape for item in arrays):
        raise ValueError("Every view must have matching (N,K) logits")
    if space == "prob":
        result = np.mean([_softmax_numpy(item) for item in arrays], axis=0)
    elif space == "logit":
        result = _softmax_numpy(np.mean(arrays, axis=0))
    else:
        raise ValueError("space must be 'prob' or 'logit'")
    return result / result.sum(axis=1, keepdims=True)


def ensemble_probs(list_of_probs):
    if not list_of_probs:
        raise ValueError("At least one probability matrix is required")
    arrays = [_as_numpy(item).astype(np.float64, copy=False) for item in list_of_probs]
    shape = arrays[0].shape
    if len(shape) != 2 or any(item.shape != shape for item in arrays):
        raise ValueError("All probability arrays must share the same (N,K) shape/order")
    if any(np.any(item < 0) for item in arrays):
        raise ValueError("Probabilities must be non-negative")
    result = np.mean(arrays, axis=0)
    denom = result.sum(axis=1, keepdims=True)
    if np.any(denom <= 0):
        raise ValueError("Probability rows must have a positive sum")
    return result / denom


def fit_temperature(val_logits, val_labels) -> float:
    """Fit one positive temperature by minimizing validation NLL."""
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("Expected logits=(N,K) and labels=(N,)")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 10.0).item())


def apply_temperature(logits, T: float):
    if not np.isfinite(T) or T <= 0:
        raise ValueError("Temperature must be finite and positive")
    return _softmax_numpy(_as_numpy(logits) / float(T))


def _temperature_scale_probs(probs, temperature: float):
    """Apply scalar temperature to probabilities without another model pass."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if temperature == 1.0:
        return _as_numpy(probs)
    log_probs = np.log(np.clip(_as_numpy(probs), 1e-12, 1.0))
    return _softmax_numpy(log_probs / float(temperature))


def fuse_conv_bn(model):
    """Return a deep-copied eval model with adjacent Conv2d/BatchNorm2d pairs fused."""
    fused_model = copy.deepcopy(model).eval()
    fuse_pair = torch.nn.utils.fusion.fuse_conv_bn_eval

    def recurse(parent):
        for child in list(parent.children()):
            recurse(child)

        children = list(parent._modules.items())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                parent._modules[conv_name] = fuse_pair(conv, bn)
                parent._modules[bn_name] = nn.Identity()

    recurse(fused_model)
    return fused_model
