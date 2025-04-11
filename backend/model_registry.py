import re
import torch.nn as nn
from collections import OrderedDict

def build_simple_cnn_backbone(model_name: str):
    backbone_classes = {
        "SimpleCNNBackbone1": SimpleCNNBackbone1,
        "SimpleCNNBackbone2": SimpleCNNBackbone2,
        "SimpleCNNBackbone3": SimpleCNNBackbone3
    }
    print(f"[DEBUG] Montando backbone: {model_name}")
    if model_name in backbone_classes:
        return backbone_classes[model_name]()

    raise ValueError(f"Backbone não reconhecido: {model_name}")


class SimpleCNNBackbone3(nn.Module):
    def __init__(self):
        super(SimpleCNNBackbone3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.out_channels = 1024

    def forward(self, x):
        return OrderedDict([('0', self.features(x))])

class SimpleCNNBackbone2(nn.Module):
    def __init__(self):
        super(SimpleCNNBackbone2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.out_channels = 256

    def forward(self, x):
        return OrderedDict([('0', self.features(x))])


class SimpleCNNBackbone1(nn.Module):
    def __init__(self):
        super(SimpleCNNBackbone1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.out_channels = 256

    def forward(self, x):
        return OrderedDict([('0', self.features(x))])