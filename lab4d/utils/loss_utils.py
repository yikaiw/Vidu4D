# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import torch
import numpy as np
import torch.nn.functional as F


def entropy_loss(prob, dim=-1):
    """Compute entropy of a probability distribution
    In the case of skinning weights, each column is a distribution over assignment to B bones.
    We want to encourage low entropy, i.e. each point is assigned to fewer bones.

    Args:
        prob: (..., B) Probability distribution
    Returns:
        entropy (...,) Entropy of each distribution
    """
    entropy = -(prob * (prob + 1e-9).log()).sum(dim)
    return entropy


def cross_entropy_skin_loss(skin):
    """Compute entropy of a probability distribution
    In the case of skinning weights, each column is a distribution over assignment to B bones.
    We want to encourage low entropy, i.e. each point is assigned to fewer bones.

    Args:
        skin: (..., B) un-normalized skinning weights
    """
    shape = skin.shape
    nbones = shape[-1]
    full_skin = skin.clone()
    # print("weight min max mean:", skin.min(), skin.max(), skin.mean())
    # find the most likely bone assignment
    score, indices = skin.max(-1, keepdim=True)
    skin = torch.zeros_like(skin).fill_(0)
    skin = skin.scatter(-1, indices, torch.ones_like(score))

    cross_entropy = F.cross_entropy(
        full_skin.view(-1, nbones), skin.view(-1, nbones), reduction="none"
    )
    cross_entropy = cross_entropy.view(shape[:-1])
    return cross_entropy


def align_vectors(v1, v2):
    """Return the scale that best aligns v1 to v2 in the L2 sense:
    min || kv1-v2 ||^2

    Args:
        v1: (...,) Source vector
        v2: (...,) Target vector
    Returns:
        scale_fac (1,): Scale factor
    """
    scale_fac = (v1 * v2).sum() / (v1 * v1).sum()
    if scale_fac < 0:
        scale_fac = torch.tensor([1.0], device=scale_fac.device)
    return scale_fac
