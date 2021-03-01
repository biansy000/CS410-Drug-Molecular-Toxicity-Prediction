import torch
import torch.nn as nn

def weighted_bce_loss(input, target, weights, size_average=True):
    input = torch.clamp(input, min=1e-12, max=1-1e-12)
    out = nn.functional.binary_cross_entropy(input, target.float(), reduction='none')
    out = out * weights

    if size_average:
        return out.sum() / weights.shape[0]
    else:
        return out.sum()


def weighted_ce_loss(input, target, weights, size_average=True):
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


def weighted_focal_loss(input, target, weights, size_average=False):
    input = torch.clamp(input, min=1e-12, max=1-1e-12).reshape(-1)
    target = target.reshape(-1)
    weights = weights.reshape(-1)
    gamma = 1.5
    # Calculate p_t
    p_t = torch.where(target==1, input, 1 - input)

    # Calculate cross entropy
    cross_entropy = -torch.log(p_t)
    weights = torch.pow((1 - p_t), gamma) * weights
    # Calculate focal loss
    loss = weights * cross_entropy
    # Sum the losses in mini_batch
    loss = torch.mean(loss)
    return loss