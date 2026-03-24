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

def fold_sequential_conv_bn(module):
    for name, child in module.named_children():
        fold_sequential_conv_bn(child)

        if isinstance(child, nn.Sequential):
            new_modules = []
            i = 0
            while i < len(child):
                m1 = child[i]

                if i + 1 < len(child):
                    m2 = child[i + 1]

                    if isinstance(m1, nn.Conv2d) and _is_supported_bn(m2):
                        print(f"[INFO] Folding Sequential Conv+BN at {name}.{i}")

                        _fold_bn_into_conv_params(m1, m2)

                        new_modules.append(m1)
                        i += 2
                        continue

                new_modules.append(m1)
                i += 1

            child[:] = nn.Sequential(*new_modules)

def count_custom_conv_with_bn(module: nn.Module):
    total = 0
    names = []
    for name, child in module.named_modules():
        if isinstance(child, Conv2d) and _is_supported_bn(getattr(child, "norm", None)):
            total += 1
            names.append(name)
    return total, names