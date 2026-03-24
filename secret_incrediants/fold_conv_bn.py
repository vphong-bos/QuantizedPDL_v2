import torch
import torch.nn as nn

from model.conv2d import Conv2d


def _is_supported_bn(norm: nn.Module) -> bool:
    return isinstance(norm, (nn.BatchNorm2d, nn.SyncBatchNorm))


def _fold_bn_into_conv_params(conv: nn.Conv2d, bn: nn.Module):
    """
    Fold BatchNorm2d or SyncBatchNorm into Conv2d parameters in-place.

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

    # General shape for Conv1d/2d/3d-style weights
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

    After folding:
        norm -> Identity
        activation stays unchanged
    """
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Recurse first
        fold_custom_conv_bn_inplace(child, full_name)

        if isinstance(child, Conv2d):
            norm = getattr(child, "norm", None)
            act = getattr(child, "activation", None)

            if norm is None or isinstance(norm, nn.Identity):
                continue

            if _is_supported_bn(norm):
                print(
                    f"[INFO] Folding custom Conv2d+BN: {full_name} "
                    f"(activation={type(act).__name__ if act is not None else 'None'})"
                )

                child.eval()
                norm.eval()

                _fold_bn_into_conv_params(child, norm)

                # Safer than None if forward always calls self.norm(x)
                child.norm = nn.Identity()
            else:
                print(
                    f"[WARN] Skip folding for {full_name}: "
                    f"norm is {type(norm)}, only BatchNorm2d and SyncBatchNorm are supported here."
                )

def count_custom_conv_with_bn(module: nn.Module):
    total = 0
    names = []
    for name, child in module.named_modules():
        if isinstance(child, Conv2d) and _is_supported_bn(getattr(child, "norm", None)):
            total += 1
            names.append(name)
    return total, names