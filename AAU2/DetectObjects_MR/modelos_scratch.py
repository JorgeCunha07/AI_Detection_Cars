import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_for_targets(tensor, targets):
    repeats = torch.tensor([len(t["labels"]) for t in targets], device=tensor.device)
    return tensor.repeat_interleave(repeats, dim=0)


# Redes Neuronais 'From Scratch' para Detecção e Atributos Globais
## Modelo 1: MultiScaleCNN
class MultiScaleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.low = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.mid = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.high = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.cls = nn.Linear(64, num_classes)
        self.box = nn.Linear(64, 4)

    def forward(self, images, targets=None):
        x = torch.stack(images)
        x = self.low(x)
        x = self.mid(x)
        x = self.high(x).view(x.size(0), -1)
        logits = self.cls(x)
        boxes = self.box(x)
        if targets is None:
            preds = torch.argmax(logits, dim=1)
            scores = torch.softmax(logits, dim=1).max(dim=1).values
            return [{"labels": p.unsqueeze(0), "boxes": b.unsqueeze(0), "scores": s.unsqueeze(0)}
                    for p, b, s in zip(preds, boxes, scores)]
        else:
            labels = torch.cat([t["labels"] for t in targets], dim=0)
            gt_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
            loss_cls = F.cross_entropy(repeat_for_targets(logits, targets), labels)
            loss_box = F.l1_loss(repeat_for_targets(boxes, targets), gt_boxes)
            return {"classification_loss": loss_cls, "bbox_regression_loss": loss_box}


## Modelo 2: TinyYOLOStyle
class TinyYOLOStyle(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.cls = nn.Linear(64, num_classes)
        self.box = nn.Linear(64, 4)

    def forward(self, images, targets=None):
        x = torch.stack(images)
        feats = self.backbone(x).view(x.size(0), -1)
        logits = self.cls(feats)
        boxes = self.box(feats)
        if targets is None:
            preds = torch.argmax(logits, dim=1)
            scores = torch.softmax(logits, dim=1).max(dim=1).values
            return [{"labels": p.unsqueeze(0), "boxes": b.unsqueeze(0), "scores": s.unsqueeze(0)}
                    for p, b, s in zip(preds, boxes, scores)]
        else:
            labels = torch.cat([t["labels"] for t in targets], dim=0)
            gt_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
            loss_cls = F.cross_entropy(repeat_for_targets(logits, targets), labels)
            loss_box = F.l1_loss(repeat_for_targets(boxes, targets), gt_boxes)
            return {"classification_loss": loss_cls, "bbox_regression_loss": loss_box}


## Modelo 3: CNN com cabeças separadas para atributos globais
class SharedTrunkMultiTask(nn.Module):
    def __init__(self, num_classes, num_weather=4, num_scene=4, num_time=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)
        self.box = nn.Linear(64, 4)
        self.weather = nn.Linear(64, num_weather)
        self.scene = nn.Linear(64, num_scene)
        self.time = nn.Linear(64, num_time)

    def forward(self, images, targets=None, attrs=None):
        x = torch.stack(images)
        f = self.features(x)
        pooled = self.pool(f).view(x.size(0), -1)
        class_logits = self.classifier(pooled)
        boxes = self.box(pooled)
        weather_logits = self.weather(pooled)
        scene_logits = self.scene(pooled)
        time_logits = self.time(pooled)

        if targets is None:
            preds = torch.argmax(class_logits, dim=1)
            scores = torch.softmax(class_logits, dim=1).max(dim=1).values
            return [{"labels": p.unsqueeze(0), "boxes": b.unsqueeze(0), "scores": s.unsqueeze(0)} for p, b, s in
                    zip(preds, boxes, scores)], {"weather": weather_logits, "scene": scene_logits, "timeofday": time_logits}
        else:
            labels = torch.cat([t["labels"] for t in targets], dim=0)
            gt_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
            loss_cls = F.cross_entropy(repeat_for_targets(class_logits, targets), labels)
            loss_box = F.l1_loss(repeat_for_targets(boxes, targets), gt_boxes)
            if attrs:
                w_true = torch.stack([a["weather"] for a in attrs]).to(pooled.device)
                s_true = torch.stack([a["scene"] for a in attrs]).to(pooled.device)
                t_true = torch.stack([a["timeofday"] for a in attrs]).to(pooled.device)
                loss_attr = F.cross_entropy(weather_logits, w_true) + F.cross_entropy(scene_logits, s_true) + F.cross_entropy(
                    time_logits, t_true)
            else:
                loss_attr = torch.tensor(0.0, device=pooled.device)
            return {"classification_loss": loss_cls, "bbox_regression_loss": loss_box, "attribute_loss": loss_attr}


## Modelo 4: CNN com atenção tipo Transformer-lite
class CNNWithAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.cls = nn.Linear(64, num_classes)
        self.box = nn.Linear(64, 4)

    def forward(self, images, targets=None):
        x = torch.stack(images)  # [B, C, H, W]
        f = self.cnn(x)  # [B, 64, H', W']
        B, C, H, W = f.shape
        tokens = f.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        attn_out, _ = self.attn(tokens, tokens, tokens)
        pooled = attn_out.mean(dim=1)  # [B, C]
        logits = self.cls(pooled)
        boxes = self.box(pooled)
        if targets is None:
            preds = torch.argmax(logits, dim=1)
            scores = torch.softmax(logits, dim=1).max(dim=1).values
            return [{"labels": p.unsqueeze(0), "boxes": b.unsqueeze(0), "scores": s.unsqueeze(0)} for p, b, s in
                    zip(preds, boxes, scores)]
        else:
            labels = torch.cat([t["labels"] for t in targets], dim=0)
            gt_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
            loss_cls = F.cross_entropy(repeat_for_targets(logits, targets), labels)
            loss_box = F.l1_loss(repeat_for_targets(boxes, targets), gt_boxes)
            return {"classification_loss": loss_cls, "bbox_regression_loss": loss_box}


## Modelo 5: CNN com Feature Pyramid Pooling
class PyramidCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        self.cls = nn.Linear(64 * (1 + 4 + 16), num_classes)
        self.box = nn.Linear(64 * (1 + 4 + 16), 4)

    def forward(self, images, targets=None):
        x = torch.stack(images)
        f = self.backbone(x)
        p1 = self.pool1(f).flatten(1)
        p2 = self.pool2(f).flatten(1)
        p3 = self.pool3(f).flatten(1)
        feats = torch.cat([p1, p2, p3], dim=1)
        logits = self.cls(feats)
        boxes = self.box(feats)
        if targets is None:
            preds = torch.argmax(logits, dim=1)
            scores = torch.softmax(logits, dim=1).max(dim=1).values
            return [{"labels": p.unsqueeze(0), "boxes": b.unsqueeze(0), "scores": s.unsqueeze(0)} for p, b, s in
                    zip(preds, boxes, scores)]
        else:
            labels = torch.cat([t["labels"] for t in targets], dim=0)
            gt_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
            loss_cls = F.cross_entropy(repeat_for_targets(logits, targets), labels)
            loss_box = F.l1_loss(repeat_for_targets(boxes, targets), gt_boxes)
            return {"classification_loss": loss_cls, "bbox_regression_loss": loss_box}