from __future__ import annotations
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights,mobilenet_v3_small

def build_queuewatch_model(pretrained:bool=False)->nn.Module:
    weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model=mobilenet_v3_small(weights=weights)
    in_features=model.classifier[-1].in_features
    model.classifier[-1]=nn.Linear(in_features,1)
    return model
