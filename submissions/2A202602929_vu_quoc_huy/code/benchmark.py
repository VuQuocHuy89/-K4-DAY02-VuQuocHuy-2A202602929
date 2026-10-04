"""Latency and throughput measurements with warmup and device synchronization."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Measure fn in milliseconds after warmup; enforce the lab's minimum counts."""
    if warmup < 10:
        raise ValueError("Use at least 10 warmup iterations")
    if iters < 50:
        raise ValueError("Use at least 50 measured iterations")
    for _ in range(warmup):
        fn()
        if sync is not None:
            sync()

    samples = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples[i] = (time.perf_counter() - start) * 1000.0

    return {
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "mean": float(samples.mean()),
        "n": int(iters),
        "warmup": int(warmup),
    }


def _measure(model, batch_size, img_size, dtype, device, warmup, iters, repeats=1):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be 'fp32', 'amp', or 'fp16'")
    if dtype == "fp16" and device.type != "cuda":
        raise ValueError("fp16 benchmarking requires a CUDA device")

    measured_model = copy.deepcopy(model).eval().to(device)
    if dtype == "fp16":
        measured_model = measured_model.half()
        input_dtype = torch.float16
    else:
        input_dtype = torch.float32
    sample = torch.randn(
        batch_size, 3, img_size, img_size, device=device, dtype=input_dtype
    )

    def forward():
        with torch.inference_mode():
            if dtype == "amp" and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    for _ in range(repeats):
                        measured_model(sample)
            else:
                for _ in range(repeats):
                    measured_model(sample)

    stats = bench(
        forward,
        warmup=warmup,
        iters=iters,
        sync=(lambda: _sync(device)) if device.type == "cuda" else None,
    )
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    stats.update({
        "gpu": gpu,
        "dtype": dtype,
        "batch": int(batch_size),
        "img_size": int(img_size),
        "views": int(repeats),
        "images_per_s": float(batch_size / (stats["p50"] / 1000.0)),
        "torch": torch.__version__,
        "device": str(device),
    })
    return stats


def latency_report(
    model,
    batch_size: int,
    img_size: int,
    dtype: str = "fp32",
    device: str = "cuda",
    warmup: int = 10,
    iters: int = 100,
) -> dict:
    """Measure one forward pass and return a row suitable for the Latency sheet."""
    return _measure(model, batch_size, img_size, dtype, device, warmup, iters, repeats=1)


def tta_latency(model, k_views: int, **kw) -> dict:
    """Measure the real latency of K sequential model forwards on one input batch."""
    if k_views < 1:
        raise ValueError("k_views must be positive")
    result = _measure(
        model,
        batch_size=kw.get("batch_size", 1),
        img_size=kw.get("img_size", 224),
        dtype=kw.get("dtype", "fp32"),
        device=kw.get("device", "cuda"),
        warmup=kw.get("warmup", 10),
        iters=kw.get("iters", 100),
        repeats=k_views,
    )
    result["method"] = f"TTA-{k_views}-view"
    return result
