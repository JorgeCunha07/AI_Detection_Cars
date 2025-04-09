import re
import torch.nn as nn
from collections import OrderedDict

def build_simple_cnn_backbone(model_name: str):
    layers = []
    in_channels = 3
    out_channels_list = [32, 64, 128, 256, 512, 1024]

    match = re.search(r"SimpleCNNBackbone(\d+)", model_name)
    if not match:
        raise ValueError(f"Nome de modelo inválido: {model_name}")
    depth = int(match.group(1))
    depth = min(depth, len(out_channels_list))  # segurança

    for i in range(depth):
        out_channels = out_channels_list[i]
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        in_channels = out_channels

    features = nn.Sequential(*layers)

    class DynamicBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.out_channels = out_channels

        def forward(self, x):
            return OrderedDict([("0", self.features(x))])

    return DynamicBackbone()
