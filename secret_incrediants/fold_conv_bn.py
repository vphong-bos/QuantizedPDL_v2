import torch
import torch.nn as nn

from model.conv2d import Conv2d


def _is_supported_bn(norm: nn.Module) -> bool:
    return isinstance(norm, (nn.BatchNorm2d, nn.SyncBatchNorm))


def _get_real_conv_module(module: nn.Module):
    if hasattr(module, "weight"):
        return module

    inner_conv = getattr(module, "conv", None)
    if inner_conv is not None and hasattr(inner_conv, "weight"):
        return inner_conv

    return None


def _describe_module(name: str, mod: nn.Module):
    norm = getattr(mod, "norm", None)
    act = getattr(mod, "activation", None)
    inner_conv = getattr(mod, "conv", None)

    print(f"[DEBUG] name={name}")
    print(f"        module type      : {type(mod)}")
    print(f"        module class     : {mod.__class__.__module__}.{mod.__class__.__name__}")
    print(f"        isinstance Conv2d: {isinstance(mod, Conv2d)}")
    print(f"        has weight       : {hasattr(mod, 'weight')}")
    print(f"        has bias         : {hasattr(mod, 'bias')}")
    print(f"        norm type        : {type(norm)}")
    print(f"        activation type  : {type(act)}")
    print(f"        has attr conv    : {inner_conv is not None}")
    if inner_conv is not None:
        print(f"        inner conv type  : {type(inner_conv)}")
        print(f"        inner has weight : {hasattr(inner_conv, 'weight')}")
        print(f"        inner has bias   : {hasattr(inner_conv, 'bias')}")


def debug_remaining_custom_conv_with_bn(module: nn.Module, max_items: int = 50):
    count = 0
    for name, child in module.named_modules():
        norm = getattr(child, "norm", None)
        if _is_supported_bn(norm):
            _describe_module(name, child)
            count += 1
            if count >= max_items:
                break


def _fold_bn_into_conv_params(conv: nn.Module, bn: nn.Module):
    if not _is_supported_bn(bn):
        raise TypeError(
            f"Unsupported BN type for folding: {type(bn)}. "
            f"Expected BatchNorm2d or SyncBatchNorm."
        )

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
    folded = 0
    skipped = 0

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        sub_folded, sub_skipped = fold_custom_conv_bn_inplace(child, full_name)
        folded += sub_folded
        skipped += sub_skipped

        norm = getattr(child, "norm", None)
        if norm is None or isinstance(norm, nn.Identity):
            continue

        if not _is_supported_bn(norm):
            continue

        real_conv = _get_real_conv_module(child)
        if real_conv is None:
            print(f"[WARN] Skip folding for {full_name}: could not find real conv module")
            _describe_module(full_name, child)
            skipped += 1
            continue

        print(
            f"[INFO] Folding: {full_name} "
            f"(module={type(child).__name__}, conv={type(real_conv).__name__}, norm={type(norm).__name__})"
        )

        child.eval()
        norm.eval()

        with torch.no_grad():
            _fold_bn_into_conv_params(real_conv, norm)

        child.norm = nn.Identity()
        folded += 1

    return folded, skipped


def count_custom_conv_with_bn(module: nn.Module):
    total = 0
    names = []
    for name, child in module.named_modules():
        norm = getattr(child, "norm", None)
        if _is_supported_bn(norm):
            total += 1
            names.append(name)
    return total, names