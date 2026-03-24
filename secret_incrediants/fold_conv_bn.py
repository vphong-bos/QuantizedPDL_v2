import torch
import torch.nn as nn

from model.conv2d import Conv2d

def _fold_bn_into_conv_params(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Fold BatchNorm2d into Conv2d parameters in-place.

    After folding:
      y = BN(Conv(x))  ==>  y = Conv_folded(x)

    Keeps checkpoint compatibility because this runs only after loading.
    """
    if conv.bias is None:
        conv_bias = torch.zeros(conv.weight.size(0), device=conv.weight.device, dtype=conv.weight.dtype)
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

    # Fold weight
    w_fold = w * inv_std.reshape([-1, 1, 1, 1])

    # Fold bias
    b_fold = beta + (b - mean) * inv_std

    conv.weight.data.copy_(w_fold)

    if conv.bias is None:
        conv.bias = nn.Parameter(b_fold.clone())
    else:
        conv.bias.data.copy_(b_fold)


def fold_custom_conv_bn_inplace(module: nn.Module, prefix: str = ""):
    """
    Recursively fold BatchNorm2d inside your custom Conv2d wrapper:
        Conv2d(..., norm=BatchNorm2d(...))
    into the conv weights/bias, then set norm=None.

    This does NOT change the original checkpoint format.
    """
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Recurse first
        fold_custom_conv_bn_inplace(child, full_name)

        if isinstance(child, Conv2d) and getattr(child, "norm", None) is not None:
            if isinstance(child.norm, nn.BatchNorm2d):
                print(f"[INFO] Folding custom Conv2d+BN: {full_name}")
                child.eval()
                child.norm.eval()

                _fold_bn_into_conv_params(child, child.norm)

                # Remove BN after folding
                child.norm = None
            else:
                print(
                    f"[WARN] Skip folding for {full_name}: "
                    f"norm is {type(child.norm)}, only BatchNorm2d is supported here."
                )

def count_custom_conv_with_bn(module: nn.Module):
    total = 0
    names = []
    for name, child in module.named_modules():
        if isinstance(child, Conv2d) and isinstance(getattr(child, "norm", None), nn.BatchNorm2d):
            total += 1
            names.append(name)
    return total, names