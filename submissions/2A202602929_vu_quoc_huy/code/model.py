"""Model construction and parameter accounting for DeepWeeds experiments."""
from __future__ import annotations

import copy

import torch
from torch import nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def get_classifier(model: nn.Module) -> nn.Module:
    """Return the model's classification head using the timm model API."""
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if classifier is None or not any(True for _ in classifier.parameters()):
        raise ValueError("Could not identify a trainable classification head")
    return classifier


def get_pretrained_tag(model: nn.Module, pretrained: bool) -> str:
    if not pretrained:
        return "random_init"
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None) or {}
    if isinstance(cfg, dict):
        return str(cfg.get("tag") or cfg.get("hf_hub_id") or cfg.get("url") or "pretrained (tag unavailable)")
    return str(cfg)


def build_model(
    name: str,
    pretrained: bool = True,
    num_classes: int = 9,
    drop_rate: float = 0.0,
    init: str = "finetune",
) -> nn.Module:
    """Create a timm classifier and replace its classifier with num_classes outputs."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Install timm before creating a model: pip install timm") from exc

    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init must be 'scratch', 'frozen', or 'finetune'")
    if num_classes != 9:
        raise ValueError("DeepWeeds has 9 classes; num_classes must be 9")

    use_pretrained = init != "scratch" and bool(pretrained)
    model = timm.create_model(
        name,
        pretrained=use_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    model._lab_pretrained_tag = get_pretrained_tag(model, use_pretrained)
    model._lab_init = init
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    classifier = get_classifier(model)
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in head_ids
    model._lab_init = "frozen"
    model._lab_frozen_head_ids = head_ids
    set_frozen_backbone_eval(model)


def set_frozen_backbone_eval(model: nn.Module) -> None:
    """Keep fully frozen submodules in eval mode after model.train()."""
    if getattr(model, "_lab_init", None) != "frozen":
        return
    for module in model.modules():
        params = list(module.parameters(recurse=True))
        if params and not any(parameter.requires_grad for parameter in params):
            # Set the flag directly so this does not recursively switch a head back to eval.
            module.training = False


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Make backbone weights, backbone norm/bias, and classifier groups."""
    classifier = get_classifier(model)
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    backbone_decay, backbone_no_decay, head_decay, head_no_decay = [], [], [], []
    seen = set()

    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if id(parameter) in head_ids:
            (head_no_decay if parameter.ndim <= 1 else head_decay).append(parameter)
        elif parameter.ndim <= 1:
            backbone_no_decay.append(parameter)
        else:
            backbone_decay.append(parameter)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0})
    if not groups:
        raise ValueError("The model has no trainable parameters")
    return groups


def count_params(model: nn.Module) -> float:
    """Return total parameter count in millions, including frozen weights."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000.0


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Estimate MACs per image with THOP; install thop for comparable reporting."""
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("Install thop to count GMACs: pip install thop") from exc

    replica = copy.deepcopy(model).eval()
    parameter = next(replica.parameters())
    device = parameter.device
    dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
    sample = torch.zeros((1, 3, img_size, img_size), device=device, dtype=dtype)
    with torch.inference_mode():
        macs, _ = profile(replica, inputs=(sample,), verbose=False)
    return float(macs) / 1e9
