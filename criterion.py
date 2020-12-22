import torch
import torch.nn as nn

def weighted_bce_loss(input, target, weights, size_average=False):
    input = torch.clamp(input, min=1e-12, max=1-1e-12)
    out = nn.functional.binary_cross_entropy(input, target.float(), reduction='none')
    out = out * weights

    if size_average:
        return out.sum() / weights.sum()
    else:
        return out.sum()


def weighted_ce_loss(input, target, weights, size_average=False):
    input = torch.clamp(input, min=1e-12, max=1-1e-12)
    input = input.reshape(-1, 2)
    target = target.reshape(-1)
    weights = weights.reshape(-1)
    out = nn.functional.cross_entropy(input, target, reduction='none')
    out = out * weights

    if size_average:
        return out.sum() / weights.sum()
    else:
        return out.sum()


def weighted_focal_loss(input, target, weights, alpha=0.5, size_average=False):
    input = torch.clamp(input, min=1e-12, max=1-1e-12)
    bce = nn.functional.binary_cross_entropy(input, target, reduction='none')
    #print(bce)
    assert (bce >= 0).all(), f'{bce}'
    pt = torch.exp(-1 * bce)

    out = bce * (1-pt+1e-8)**alpha * weights

    if size_average:
        return out.sum() / weights.sum()
    else:
        return out.sum()