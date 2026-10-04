"""Shared training loop for backbone, recipe and final experiments."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    import dataset
    import inference
    import losses
    import model as model_lib
except ImportError:  # package import, e.g. from code.train import run
    from . import dataset, inference, losses, model as model_lib


@dataclass
class Config:
    # Identity
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # Model
    backbone: str = "resnet50"
    init: str = "finetune"             # scratch | frozen | finetune
    drop_rate: float = 0.0
    # Data and augmentation
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None         # None | balanced
    mix: str | None = None             # None | mixup | cutmix
    mix_alpha: float = 1.0
    # Loss
    loss: str = "ce"                   # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # Optimization
    optimizer: str = "adamw"           # adamw | sgd
    momentum: float = 0.9
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    gradient_clip: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # Paths
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curve_dir: str = "curves"
    # Test predictions must only be enabled for the final evaluation.
    save_test_predictions: bool = False
    inference_method: str = "I00"
    inference_aggregation: str = "prob"
    temperature: float = 1.0
    inference_crop_size: int = 224
    inference_scales: tuple[int, ...] = (224, 256)
    resume: bool = True


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_eval_api():
    try:
        import eval as eval_api
        return eval_api
    except ImportError:
        # In the required submission layout, train.py is under repo/submissions/.../code.
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import eval as eval_api
        return eval_api


def build_optimizer(model, cfg: Config):
    groups = model_lib.param_groups(
        model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay
    )
    name = cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=cfg.momentum, nesterov=True)
    raise ValueError("optimizer must be 'adamw' or 'sgd'")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = min(total_steps, max(0, int(cfg.warmup_epochs * steps_per_epoch)))

    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class EMA:
    """Exponential moving average of parameters and model buffers."""

    def __init__(self, model, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between 0 and 1")
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model) -> None:
        current = model.state_dict()
        for key, value in current.items():
            shadow = self.shadow[key]
            value = value.detach()
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow, strict=True)


def _autocast(device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _make_scaler(enabled: bool):
    enabled = bool(enabled and torch.cuda.is_available())
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _criterion(cfg: Config, train_df):
    label_counts = (
        train_df["Label"].value_counts()
        .reindex(range(dataset.NUM_CLASSES), fill_value=0)
        .to_dict()
    )
    if cfg.loss == "ce_weighted":
        beta = 0.0 if cfg.class_weight_beta is None else cfg.class_weight_beta
        weight = losses.class_weights(label_counts, beta=beta)
        return losses.build_criterion("ce_weighted", weight=weight)
    if cfg.loss == "focal":
        alpha = None
        if cfg.class_weight_beta is not None:
            alpha = losses.class_weights(label_counts, beta=cfg.class_weight_beta)
        return losses.build_criterion("focal", gamma=cfg.focal_gamma, alpha=alpha)
    if cfg.loss == "ls":
        return losses.build_criterion("ls", smoothing=cfg.label_smoothing)
    if cfg.loss == "ce":
        return losses.build_criterion("ce")
    raise ValueError(f"Unsupported loss: {cfg.loss}")


def _metrics(eval_api, filenames, y_true, logits):
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    result = eval_api.compute_metrics(y_true, probs.argmax(1), probs)
    return result, probs, logits


def evaluate(model, loader, criterion, device):
    """Return filenames, labels, logits, and mean loss without gradients."""
    model.eval()
    all_names, all_labels, all_logits = [], [], []
    loss_sum, seen = 0.0, 0
    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits.float(), labels)
            batch = labels.shape[0]
            loss_sum += float(loss.item()) * batch
            seen += batch
            all_names.extend(list(filenames))
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.float().cpu().numpy())

    if not seen:
        raise RuntimeError("Evaluation loader produced no batches")
    return (
        all_names,
        np.concatenate(all_labels),
        np.concatenate(all_logits),
        loss_sum / seen,
    )


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device, ema: EMA | None = None) -> dict:
    model.train()
    if cfg.init == "frozen":
        model_lib.set_frozen_backbone_eval(model)

    total_loss, seen = 0.0, 0
    lr_values = []
    for images, labels, _filenames in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = labels
        if cfg.mix:
            images, targets = losses.mix_batch(
                images, labels, alpha=cfg.mix_alpha, mode=cfg.mix
            )

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, cfg.amp):
            logits = model(images)
            loss = losses.mixed_loss(criterion, logits.float(), targets)

        scaler.scale(loss).backward()
        if cfg.gradient_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.gradient_clip,
            )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        batch = labels.shape[0]
        total_loss += float(loss.detach().item()) * batch
        seen += batch
        lr_values.append(float(optimizer.param_groups[0]["lr"]))

    if not seen:
        raise RuntimeError("Training loader produced no batches")
    return {
        "train_loss": total_loss / seen,
        "lr": float(np.mean(lr_values)) if lr_values else 0.0,
    }


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("Cannot plot an empty history")
    frame = pd.DataFrame(history)
    epochs = frame["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, frame["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, frame["val_loss"], marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, frame["val_macro_f1"], marker="o", label="val macro-F1")
    axes[1].plot(epochs, frame["val_top1"], marker="o", label="val top-1")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _to_json(value):
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def run(cfg: Config) -> dict:
    """Train one experiment; test is loaded/evaluated only when explicitly enabled."""
    if cfg.epochs < 1:
        raise ValueError("epochs must be at least 1")
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_api = _resolve_eval_api()
    output_dir = run_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    resume_checkpoint_exists = bool(cfg.resume and last_path.is_file())
    curve_path = Path(cfg.curve_dir) / f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png"
    prediction_dir = Path(cfg.pred_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    config_record = dataclasses.asdict(cfg)
    config_record["device"] = str(device)
    config_record["torch_version"] = torch.__version__

    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    split_report = dataset.check_split(train_df, val_df, test_df, cfg.images_dir, fold=cfg.fold)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir,
        dataset.build_transforms(True, cfg.img_size, cfg.aug),
        cfg.batch_size, train=True, sampler=cfg.sampler,
        num_workers=cfg.num_workers,
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir,
        dataset.build_transforms(False, cfg.img_size),
        cfg.batch_size, train=False, sampler=None,
        num_workers=cfg.num_workers,
    )
    test_loader = None
    if cfg.save_test_predictions:
        test_img_size = max(
            cfg.img_size,
            cfg.inference_crop_size + 32 if cfg.inference_method.upper() == "I02" else cfg.img_size,
        )
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir,
            dataset.build_transforms(False, test_img_size),
            cfg.batch_size, train=False, sampler=None,
            num_workers=cfg.num_workers,
        )

    network = model_lib.build_model(
        cfg.backbone, pretrained=(cfg.init != "scratch" and not resume_checkpoint_exists),
        num_classes=dataset.NUM_CLASSES, drop_rate=cfg.drop_rate, init=cfg.init,
    ).to(device)
    criterion = _criterion(cfg, train_df).to(device)
    optimizer = build_optimizer(network, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = _make_scaler(cfg.amp)
    ema = EMA(network, cfg.ema_decay) if cfg.ema_decay is not None else None
    ema_model = None
    if ema is not None:
        import copy
        ema_model = copy.deepcopy(network).eval()

    try:
        gmacs = model_lib.count_gmacs(network, cfg.img_size)
        gmacs_error = None
    except Exception as exc:
        gmacs, gmacs_error = None, f"{type(exc).__name__}: {exc}"

    parameter_millions = model_lib.count_params(network)
    pretrained_tag = getattr(network, "_lab_pretrained_tag", "unknown")
    history, best_state, best_epoch, best_macro_f1 = [], None, None, -float("inf")
    epoch_seconds = []
    start_epoch = 1
    if resume_checkpoint_exists:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("config", {})
        if any(saved_config.get(key) != value for key, value in dataclasses.asdict(cfg).items()):
            raise ValueError(
                f"Cannot resume {last_path}: saved configuration differs from the requested one. "
                "Use a new exp_id or remove that run directory."
            )
        network.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if ema is not None and checkpoint.get("ema_shadow") is not None:
            ema.shadow = {
                key: value.to(device) for key, value in checkpoint["ema_shadow"].items()
            }
        history = checkpoint["history"]
        best_state = checkpoint["best_state"]
        best_epoch = checkpoint["best_epoch"]
        best_macro_f1 = checkpoint["best_macro_f1"]
        pretrained_tag = checkpoint.get("pretrained_tag", pretrained_tag)
        epoch_seconds = [row["epoch_seconds"] for row in history]
        start_epoch = int(checkpoint["epoch"]) + 1
        if checkpoint.get("loader_generator_state") is not None:
            train_loader.generator.set_state(checkpoint["loader_generator_state"])
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"Resuming {cfg.exp_id} at epoch {start_epoch}/{cfg.epochs}")
    (output_dir / "config.json").write_text(
        json.dumps(_to_json(config_record), indent=2), encoding="utf-8"
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        start = time.perf_counter()
        train_stats = train_one_epoch(
            network, train_loader, criterion, optimizer, scheduler, scaler,
            cfg, device, ema=ema,
        )
        evaluated_model = network
        if ema is not None:
            ema.copy_to(ema_model)
            model_lib.set_frozen_backbone_eval(ema_model)
            evaluated_model = ema_model
        val_names, val_labels, val_logits, val_loss = evaluate(
            evaluated_model, val_loader, criterion, device
        )
        val_metrics, _val_probs, _ = _metrics(
            eval_api, val_names, val_labels, val_logits
        )
        elapsed = time.perf_counter() - start
        epoch_seconds.append(elapsed)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_top1": val_metrics["top1"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_ece": val_metrics["ece"],
            "lr": train_stats["lr"],
            "epoch_seconds": elapsed,
        }
        history.append(row)
        print(
            f"{cfg.exp_id} seed={cfg.seed} epoch={epoch}/{cfg.epochs} "
            f"train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} val_top1={row['val_top1']:.4f} "
            f"seconds={elapsed:.1f}"
        )

        # Strict comparison keeps the earlier epoch on ties.
        if row["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = row["val_macro_f1"]
            best_epoch = epoch
            source_model = evaluated_model
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in source_model.state_dict().items()
            }
            torch.save(
                {
                    "state_dict": best_state,
                    "config": config_record,
                    "best_epoch": best_epoch,
                    "pretrained_tag": pretrained_tag,
                },
                output_dir / "best.pt",
            )

        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        last_checkpoint = {
            "state_dict": {key: value.detach().cpu() for key, value in network.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "ema_shadow": None if ema is None else {
                key: value.detach().cpu() for key, value in ema.shadow.items()
            },
            "epoch": epoch,
            "history": history,
            "best_state": best_state,
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
            "pretrained_tag": pretrained_tag,
            "config": dataclasses.asdict(cfg),
            "loader_generator_state": train_loader.generator.get_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary_last_path = last_path.with_suffix(".tmp")
        torch.save(last_checkpoint, temporary_last_path)
        temporary_last_path.replace(last_path)

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    plot_curves(history, curve_path, f"{cfg.exp_id} — {cfg.backbone} — seed {cfg.seed}")

    network.load_state_dict(best_state)
    network.to(device)
    val_names, val_labels, val_logits, val_loss = evaluate(
        network, val_loader, criterion, device
    )
    val_metrics, val_probs, val_logits = _metrics(
        eval_api, val_names, val_labels, val_logits
    )
    np.savez_compressed(
        output_dir / "val_logits.npz",
        filenames=np.asarray(val_names),
        y_true=val_labels,
        logits=val_logits,
    )
    final_val_names, final_val_labels, final_val_probs = val_names, val_labels, val_probs
    if cfg.save_test_predictions:
        # The selected inference method is applied to validation for matching
        # reporting and then to test exactly once below.
        final_val_img_size = max(
            cfg.img_size,
            cfg.inference_crop_size + 32 if cfg.inference_method.upper() == "I02" else cfg.img_size,
        )
        if final_val_img_size == cfg.img_size:
            final_val_loader = val_loader
        else:
            final_val_loader = dataset.make_loader(
                val_df, cfg.images_dir,
                dataset.build_transforms(False, final_val_img_size),
                cfg.batch_size, train=False, sampler=None,
                num_workers=cfg.num_workers,
            )
        final_val_names, final_val_labels, final_val_probs = inference.predict_probabilities(
            network,
            final_val_loader,
            device,
            method=cfg.inference_method,
            temperature=cfg.temperature,
            aggregation=cfg.inference_aggregation,
            crop_size=cfg.inference_crop_size,
            scales=cfg.inference_scales,
        )
        np.savez_compressed(
            output_dir / "val_final_predictions.npz",
            filenames=np.asarray(final_val_names),
            y_true=final_val_labels,
            probabilities=final_val_probs,
        )
    eval_api.save_predictions(
        pred_path(cfg, "val"), final_val_names, final_val_labels, final_val_probs
    )
    final_val_metrics = eval_api.compute_metrics(
        final_val_labels, final_val_probs.argmax(axis=1), final_val_probs
    )

    result = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "init": cfg.init,
        "pretrained_tag": pretrained_tag,
        "best_epoch": best_epoch,
        "epochs": cfg.epochs,
        "macro_f1_val": final_val_metrics["macro_f1"],
        "top1_val": final_val_metrics["top1"],
        "balanced_acc_val": final_val_metrics["balanced_acc"],
        "ece_val": final_val_metrics["ece"],
        "val_loss": val_loss,
        "params_m": parameter_millions,
        "gmacs": gmacs,
        "gmacs_error": gmacs_error,
        "mean_epoch_seconds": float(np.mean(epoch_seconds)),
        "curve": str(curve_path),
        "checkpoint": str(output_dir / "best.pt"),
        "split": split_report,
        "history": history,
    }

    if test_loader is not None:
        # This branch is only enabled for final runs after the configuration is frozen.
        test_names, test_labels, test_probs, test_uncalibrated_probs = inference.predict_probabilities(
            network,
            test_loader,
            device,
            method=cfg.inference_method,
            temperature=cfg.temperature,
            aggregation=cfg.inference_aggregation,
            crop_size=cfg.inference_crop_size,
            scales=cfg.inference_scales,
            return_uncalibrated=True,
        )
        test_metrics = eval_api.compute_metrics(
            test_labels, test_probs.argmax(axis=1), test_probs
        )
        np.savez_compressed(
            output_dir / "test_predictions.npz",
            filenames=np.asarray(test_names),
            y_true=test_labels,
            probabilities=test_probs,
        )
        eval_api.save_predictions(
            pred_path(cfg, "test"), test_names, test_labels, test_probs
        )
        eval_api.save_predictions(
            Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_test_uncal.csv",
            test_names, test_labels, test_uncalibrated_probs,
        )
        result["test"] = {
            "macro_f1": test_metrics["macro_f1"],
            "top1": test_metrics["top1"],
            "balanced_acc": test_metrics["balanced_acc"],
            "ece": test_metrics["ece"],
            "nll": test_metrics["nll"],
            "precision": test_metrics["precision"].tolist(),
            "recall": test_metrics["recall"].tolist(),
            "f1": test_metrics["f1"].tolist(),
            "confusion": test_metrics["confusion"].tolist(),
        }

    (output_dir / "summary.json").write_text(
        json.dumps(_to_json(result), indent=2), encoding="utf-8"
    )
    print(
        f"Best {cfg.exp_id} seed={cfg.seed}: epoch={best_epoch}, "
        f"macro-F1 val={final_val_metrics['macro_f1']:.4f}, "
        f"top-1 val={final_val_metrics['top1']:.4f}"
    )
    return result


def parse_overrides(pairs: list[str]) -> dict:
    defaults = Config()
    values = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        if not hasattr(defaults, key):
            raise ValueError(f"Unknown Config field: {key}")
        default = getattr(defaults, key)
        lowered = raw.lower()
        if lowered in {"none", "null"}:
            value = None
        elif isinstance(default, bool):
            if lowered not in {"true", "false", "1", "0", "yes", "no"}:
                raise ValueError(f"{key} expects a boolean, got {raw!r}")
            value = lowered in {"true", "1", "yes"}
        elif isinstance(default, int) and not isinstance(default, bool):
            value = int(raw)
        elif isinstance(default, float):
            value = float(raw)
        elif isinstance(default, tuple):
            value = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
        elif default is None:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        else:
            value = raw
        values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DeepWeeds experiment")
    parser.add_argument("--set", nargs="+", required=True, metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(dataclasses.asdict(cfg), indent=2))
    result = run(cfg)
    print(json.dumps(_to_json(result), indent=2))


if __name__ == "__main__":
    main()
