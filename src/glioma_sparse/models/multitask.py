"""
Stage A network with an auxiliary molecular-subtype head.

The grade network is a standard torchvision ResNet whose `fc` is the 3-class
grade head, exactly as built by `build_model`. The auxiliary head is a separate
linear layer over the same pooled backbone features. Keeping the two apart is
deliberate: `net.state_dict()` stays loadable by every existing script (patch
injection, cohort building, end-to-end inference, dashboard), and the auxiliary
head is stored and loaded on its own. The grade head therefore drives the
patch-injection risk map exactly as before; the auxiliary head only shapes the
shared features during training and can optionally supply a slide-level subtype
estimate at inference.
"""

import torch
import torch.nn as nn

from glioma_sparse.models.factory import build_model


def pooled_features(net, x):
    """Pooled (N, F) backbone features of a torchvision ResNet."""
    x = net.conv1(x)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)
    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4(x)
    return torch.flatten(net.avgpool(x), 1)


class GradeAuxNet(nn.Module):
    """ResNet grade classifier + auxiliary subtype head on shared features."""

    def __init__(self, model_name, n_grade=3, n_subtype=3, pretrained=False):
        super().__init__()
        self.net = build_model(model_name, num_classes=n_grade, pretrained=pretrained)
        self.aux = nn.Linear(self.net.fc.in_features, n_subtype)

    def forward(self, x):
        f = pooled_features(self.net, x)
        return self.net.fc(f), self.aux(f)

    # --- checkpoint IO: grade net stays a plain ResNet state dict ---
    def load_grade(self, path, map_location="cpu"):
        self.net.load_state_dict(
            torch.load(path, map_location=map_location, weights_only=True))

    def load_aux(self, path, map_location="cpu"):
        self.aux.load_state_dict(
            torch.load(path, map_location=map_location, weights_only=True))

    def save(self, grade_path, aux_path):
        torch.save(self.net.state_dict(), grade_path)
        torch.save(self.aux.state_dict(), aux_path)


def build_aux_head(model_name, n_subtype=3):
    """Standalone auxiliary head (for inference with a separately loaded net)."""
    in_f = build_model(model_name, num_classes=3, pretrained=False).fc.in_features
    return nn.Linear(in_f, n_subtype)


@torch.no_grad()
def subtype_probs(net, aux_head, x):
    """Softmax subtype probabilities for a batch, from a plain ResNet + aux head."""
    return torch.softmax(aux_head(pooled_features(net, x)), dim=1)
