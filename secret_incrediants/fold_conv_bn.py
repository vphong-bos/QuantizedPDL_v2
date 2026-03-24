import torch
import torch.nn as nn

import copy
import torch.nn as nn



def is_supported_bn(norm):
    return isinstance(norm, (nn.BatchNorm2d, nn.SyncBatchNorm))


def fold_conv_bn_params(conv_weight, conv_bias, bn):
    if conv_bias is None:
        conv_bias = torch.zeros(
            conv_weight.shape[0],
            device=conv_weight.device,
            dtype=conv_weight.dtype,
        )

    gamma = bn.weight if bn.affine else torch.ones_like(bn.running_mean)
    beta = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    denom = torch.sqrt(var + eps)
    scale = gamma / denom

    view_shape = [conv_weight.shape[0]] + [1] * (conv_weight.dim() - 1)
    folded_weight = conv_weight * scale.reshape(view_shape)
    folded_bias = beta + (conv_bias - mean) * scale

    return folded_weight, folded_bias


def fold_bn_into_conv_module(conv_module, bn_module):
    with torch.no_grad():
        folded_weight, folded_bias = fold_conv_bn_params(
            conv_module.weight,
            conv_module.bias,
            bn_module,
        )
        conv_module.weight.copy_(folded_weight)

        if conv_module.bias is None:
            conv_module.bias = nn.Parameter(folded_bias.clone())
        else:
            conv_module.bias.copy_(folded_bias)

def convert_syncbn_to_bn(module: nn.Module) -> nn.Module:
    module = copy.deepcopy(module)

    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            new_bn = nn.BatchNorm2d(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )

            if child.affine:
                new_bn.weight.data.copy_(child.weight.data)
                new_bn.bias.data.copy_(child.bias.data)

            if child.track_running_stats:
                new_bn.running_mean.data.copy_(child.running_mean.data)
                new_bn.running_var.data.copy_(child.running_var.data)
                new_bn.num_batches_tracked.data.copy_(child.num_batches_tracked.data)

            setattr(module, name, new_bn)
        else:
            setattr(module, name, convert_syncbn_to_bn(child))

    return module

def count_custom_conv_with_bn(module: nn.Module):
    total = 0
    names = []
    for name, child in module.named_modules():
        if isinstance(child, Conv2d) and isinstance(getattr(child, "norm", None), nn.BatchNorm2d):
            total += 1
            names.append(name)
    return total, names