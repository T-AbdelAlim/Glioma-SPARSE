"""Test-time augmentation over the eight dihedral views (4 rotations x mirror)."""

import torch


def dihedral_views(x):
    """The 8 dihedral views of a (N, C, H, W) batch (identity first)."""
    views = []
    for flip in (False, True):
        base = torch.flip(x, dims=[3]) if flip else x
        for k in range(4):
            views.append(torch.rot90(base, k, dims=[2, 3]))
    return views


@torch.no_grad()
def predict_probs(model, x, tta=False):
    """Softmax probabilities for a batch; with tta, the mean over 8 views."""
    if not tta:
        return torch.softmax(model(x), dim=1)
    return torch.stack(
        [torch.softmax(model(v), dim=1) for v in dihedral_views(x)]).mean(0)
