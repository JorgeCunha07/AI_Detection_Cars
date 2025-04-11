import torch
import torch.nn as nn
from torch.utils.data import Dataset

###########################################
# 3. Classe MultiTaskModel (Detecção + Atributos Globais)
###########################################
class MultiTaskModel(nn.Module):
    def __init__(self, detection_model, backbone, num_weather, num_scene, num_time, in_features):
        super(MultiTaskModel, self).__init__()
        self.detection_model = detection_model
        self.backbone = backbone
        self.attr_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_weather = nn.Linear(in_features, num_weather)
        self.fc_scene = nn.Linear(in_features, num_scene)
        self.fc_timeofday = nn.Linear(in_features, num_time)
    def forward(self, images, targets=None, global_attrs=None):
        # Em treinamento, retorna losses; em avaliação, retorna as detecções
        if self.training:
            detection_loss = self.detection_model(images, targets)
        else:
            detection_loss = self.detection_model(images)
        imgs_tensor = torch.stack(images)
        feats = self.backbone(imgs_tensor)
        pooled = self.attr_pool(feats)
        pooled = pooled.view(pooled.size(0), -1)
        weather_logits = self.fc_weather(pooled)
        scene_logits = self.fc_scene(pooled)
        timeofday_logits = self.fc_timeofday(pooled)
        if self.training and global_attrs is not None:
            weather_labels = torch.stack([attr["weather"] for attr in global_attrs]).to(images[0].device)
            scene_labels = torch.stack([attr["scene"] for attr in global_attrs]).to(images[0].device)
            timeofday_labels = torch.stack([attr["timeofday"] for attr in global_attrs]).to(images[0].device)
            loss_weather = nn.functional.cross_entropy(weather_logits, weather_labels)
            loss_scene = nn.functional.cross_entropy(scene_logits, scene_labels)
            loss_timeofday = nn.functional.cross_entropy(timeofday_logits, timeofday_labels)
            attr_loss = loss_weather + loss_scene + loss_timeofday
        else:
            attr_loss = 0
        return detection_loss, attr_loss, weather_logits, scene_logits, timeofday_logits