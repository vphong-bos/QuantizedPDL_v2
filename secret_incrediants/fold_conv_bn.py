import torch
import torch.nn as nn

from model.conv2d import Conv2d


def _is_supported_bn(norm: nn.Module) -> bool:
    return isinstance(norm, (nn.BatchNorm2d, nn.SyncBatchNorm))


def _get_real_conv_module(module: nn.Module):
    """
    Return the actual Conv module that owns weight/bias.

    Supports:
    - custom Conv2d subclassing nn.Conv2d directly
    - wrapper module with .conv holding nn.Conv2d
    """
    if hasattr(module, "weight") and hasattr(module, "bias"):
        return module

    inner_conv = getattr(module, "conv", None)
    if inner_conv is not None and hasattr(inner_conv, "weight") and hasattr(inner_conv, "bias"):
        return inner_conv

    return None


def _fold_bn_into_conv_params(conv: nn.Module, bn: nn.Module):
    """
    Fold BatchNorm2d or SyncBatchNorm into Conv parameters in-place.

    Valid for:
        Conv -> BN
        Conv -> BN -> ReLU
        Conv -> BN -> other activation

    After folding:
      y = Act(BN(Conv(x)))  ==>  y = Act(Conv_folded(x))
    """
    if not _is_supported_bn(bn):
        raise TypeError(
            f"Unsupported BN type for folding: {type(bn)}. "
            f"Expected BatchNorm2d or SyncBatchNorm."
        )

    if conv is None or not hasattr(conv, "weight"):
        raise TypeError("conv must be a Conv-like module with weight/bias")

    if conv.bias is None:
        conv_bias = torch.zeros(
            conv.weight.size(0),
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
    else:
        conv_bias = conv.bias.data

    w = conv.weight.data
    b = conv_bias

    gamma = bn.weight.data if bn.affine else torch.ones_like(bn.running_mean)
    beta = bn.bias.data if bn.affine else torch.zeros_like(bn.running_mean)
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = bn.eps

    inv_std = gamma / torch.sqrt(var + eps)

    reshape_dims = [-1] + [1] * (w.dim() - 1)

    w_fold = w * inv_std.reshape(reshape_dims)
    b_fold = beta + (b - mean) * inv_std

    conv.weight.data.copy_(w_fold)

    if conv.bias is None:
        conv.bias = nn.Parameter(b_fold.clone())
    else:
        conv.bias.data.copy_(b_fold)


def fold_custom_conv_bn_inplace(module: nn.Module, prefix: str = ""):
    """
    Recursively fold BN/SyncBN inside custom Conv2d wrapper.

    Supports patterns like:
        Conv2d(norm=BatchNorm2d, activation=ReLU)
        Conv2d(norm=SyncBatchNorm, activation=ReLU)

    Also supports wrapper-style modules where the real conv lives in:
        child.conv

    After folding:
        norm -> Identity
        activation stays unchanged
    """
    folded = 0
    skipped = 0

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Recurse first
        sub_folded, sub_skipped = fold_custom_conv_bn_inplace(child, full_name)
        folded += sub_folded
        skipped += sub_skipped

        if not isinstance(child, Conv2d):
            continue

        norm = getattr(child, "norm", None)
        act = getattr(child, "activation", None)

        if norm is None or isinstance(norm, nn.Identity):
            continue

        if not _is_supported_bn(norm):
            print(
                f"[WARN] Skip folding for {full_name}: "
                f"norm is {type(norm)}, only BatchNorm2d and SyncBatchNorm are supported here."
            )
            skipped += 1
            continue

        real_conv = _get_real_conv_module(child)
        if real_conv is None:
            print(
                f"[WARN] Skip folding for {full_name}: "
                f"could not find real conv module with weight/bias "
                f"(child type={type(child)})"
            )
            skipped += 1
            continue

        print(
            f"[INFO] Folding custom Conv2d+BN: {full_name} "
            f"(activation={type(act).__name__ if act is not None else 'None'}, "
            f"real_conv={type(real_conv).__name__})"
        )

        child.eval()
        norm.eval()

        _fold_bn_into_conv_params(real_conv, norm)

        # Keep forward safe even if code always calls self.norm(x)
        child.norm = nn.Identity()
        folded += 1

    return folded, skipped


def count_custom_conv_with_bn(module: nn.Module):
    total = 0
    names = []
    for name, child in module.named_modules():
        if isinstance(child, Conv2d) and _is_supported_bn(getattr(child, "norm", None)):
            total += 1
            names.append(name)
    return total, names


def debug_custom_conv_structure(module: nn.Module, match_substrings=None):
    """
    Optional helper to inspect remaining unfused custom conv blocks.
    """
    match_substrings = match_substrings or []

    for name, child in module.named_modules():
        if not isinstance(child, Conv2d):
            continue

        if match_substrings and not any(s in name for s in match_substrings):
            continue

        norm = getattr(child, "norm", None)
        act = getattr(child, "activation", None)
        real_conv = _get_real_conv_module(child)

        print(f"[DEBUG] {name}")
        print(f"        child type      : {type(child)}")
        print(f"        real_conv type  : {type(real_conv) if real_conv is not None else None}")
        print(f"        norm type       : {type(norm)}")
        print(f"        activation type : {type(act)}")
        if real_conv is not None:
            print(f"        has weight      : {hasattr(real_conv, 'weight')}")
            print(f"        has bias        : {hasattr(real_conv, 'bias')}")