import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def get_dataset_params():
    import json
    with open("helpers/categories.json", "r") as f:
        cat_map = json.load(f)
    with open("helpers/weather.json", "r") as f:
        weather_map = json.load(f)
    with open("helpers/scene.json", "r") as f:
        scene_map = json.load(f)
    with open("helpers/timeofday.json", "r") as f:
        time_map = json.load(f)

    num_classes = len(cat_map) + 1
    num_weather = len(weather_map)
    num_scene = len(scene_map)
    num_time = len(time_map)

    return num_classes, num_weather, num_scene, num_time


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
        )
        self.out_channels = 128

    def forward(self, x):
        return self.body(x)


class MidCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU()
        )
        self.out_channels = 256

    def forward(self, x):
        return self.body(x)


class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU()
        )
        self.out_channels = 512

    def forward(self, x):
        return self.body(x)


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.out_channels = module.out_channels

    def forward(self, x):
        if isinstance(x, list):  # esperado em treino
            x = torch.stack(x)
        return {"0": self.module(x)}


class FasterRCNNMultiTask(FasterRCNN):
    def __init__(self, backbone_type):
        num_classes, num_weather, num_scene, num_time = get_dataset_params()

        if backbone_type == "simplecnn":
            custom = SimpleCNN()
        elif backbone_type == "midcnn":
            custom = MidCNN()
        elif backbone_type == "deepcnn":
            custom = DeepCNN()
        else:
            raise ValueError(f"Backbone não reconhecido: {backbone_type}")

        backbone = Wrapper(custom)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )

        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        self.custom_backbone = custom
        self.attr_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_weather = nn.Linear(custom.out_channels, num_weather)
        self.fc_scene = nn.Linear(custom.out_channels, num_scene)
        self.fc_timeofday = nn.Linear(custom.out_channels, num_time)

    def forward(self, images, targets=None, attrs=None):
        if self.training and targets is None:
            raise ValueError("Targets devem ser fornecidos durante o treino")

        # Extração de features
        features = self.backbone(images)["0"]
        pooled = self.attr_pool(features).flatten(1)

        attr_logits = {
            "weather": self.fc_weather(pooled),
            "scene": self.fc_scene(pooled),
            "timeofday": self.fc_timeofday(pooled)
        }

        # Forward do Faster R-CNN
        output = super().forward(images, targets)

        if self.training:
            weather_gt = torch.stack([a["weather"] for a in attrs]).to(features.device)
            scene_gt = torch.stack([a["scene"] for a in attrs]).to(features.device)
            time_gt = torch.stack([a["timeofday"] for a in attrs]).to(features.device)

            output["loss_weather"]   = nn.functional.cross_entropy(attr_logits["weather"], weather_gt)
            output["loss_scene"]     = nn.functional.cross_entropy(attr_logits["scene"], scene_gt)
            output["loss_timeofday"] = nn.functional.cross_entropy(attr_logits["timeofday"], time_gt)
            return output
        else:
            return output, attr_logits


def build_fasterrcnn_model(backbone_type="simplecnn"):
    return FasterRCNNMultiTask(backbone_type)
