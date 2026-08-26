import torch.nn as nn
from torchvision import models

_CTOR = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50}
_WEIGHTS = {"resnet18": models.ResNet18_Weights.DEFAULT,
           "resnet34": models.ResNet34_Weights.DEFAULT,
           "resnet50": models.ResNet50_Weights.DEFAULT}


def build_model(name, num_classes, pretrained=True):
    if name not in _CTOR:
        raise ValueError("Unknown model: {}".format(name))

    model = _CTOR[name](weights=_WEIGHTS[name] if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model