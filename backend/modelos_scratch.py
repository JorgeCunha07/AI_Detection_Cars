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

def SimpleCNN():
    backbone = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()
    )
    backbone.out_channels = 128
    return backbone

def MidCNN():
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU()
    )
    backbone.out_channels = 256
    return backbone

def DeepCNN():
    backbone = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU()
    )
    backbone.out_channels = 512
    return backbone

def build_fasterrcnn_model(backbone_type="simplecnn"):
    num_classes, _, _, _ = get_dataset_params()

    if backbone_type == "simplecnn":
        backbone = SimpleCNN()
    elif backbone_type == "midcnn":
        backbone = MidCNN()
    elif backbone_type == "deepcnn":
        backbone = DeepCNN()
    else:
        raise ValueError(f"Backbone não reconhecido: {backbone_type}")

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model
