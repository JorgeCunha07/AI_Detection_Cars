# models.py
import os
import re
import torch
import yaml
import base64
import tempfile
import cv2
from PIL import Image
from collections import Counter
from torchvision import transforms
from ultralytics import YOLO
import base64


def load_class_names_from_yaml(yaml_path="dataset.yaml"):
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return {i: name for i, name in enumerate(data.get("names", []))}


def encode_image_base64(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode("utf-8")


def detect_labels_DataSet1(model_name: str, base64_image: str):
    model_path = os.path.join("modelsAvailable/1", f"{model_name}.pt")
    yaml_path = "dataset.yaml"

    if not os.path.exists(model_path):
        return {"error": f"Modelo '{model_name}' não encontrado."}, 404

    class_map = load_class_names_from_yaml(yaml_path)

    # Decode do base64 para bytes
    try:
        image_bytes = base64.b64decode(base64_image)
    except Exception as e:
        return {"error": f"Imagem base64 inválida: {str(e)}"}, 400

    # Salvar temporariamente
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(image_bytes)
    temp.close()

    label_counts = {}
    image_bgr = cv2.imread(temp.name)

    if re.search(r"yolo", model_name, re.IGNORECASE):
        yolo_model = YOLO(model_path)
        results = yolo_model(temp.name)[0]

        class_ids = [int(cls) for cls in results.boxes.cls]
        label_names = [class_map.get(class_id, f"class_{class_id}") for class_id in class_ids]
        label_counts = dict(Counter(label_names))

        for box, cls in zip(results.boxes.xyxy, class_ids):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = class_map.get(cls, f"class_{cls}")
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elif re.search(r"rcnn|mobilenet|vgg", model_name, re.IGNORECASE):
        image = Image.open(temp.name).convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image)

        rcnn_model = torch.load(model_path, map_location="cpu")
        rcnn_model.eval()

        with torch.no_grad():
            outputs = rcnn_model([image_tensor])[0]

        min_score = 0.5
        for i in range(len(outputs["boxes"])):
            score = outputs["scores"][i].item()
            if score >= min_score:
                box = outputs["boxes"][i].tolist()
                class_id = int(outputs["labels"][i].item())
                label = class_map.get(class_id, f"class_{class_id}")
                label_counts[label] = label_counts.get(label, 0) + 1

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    else:
        os.remove(temp.name)
        return {"error": "Tipo de modelo não reconhecido no nome."}, 400

    image_base64 = encode_image_base64(image_bgr)
    os.remove(temp.name)

    return {
        "labels": list(label_counts.keys()),
        "counts": label_counts,
        "image_base64": image_base64
    }, 200


def detect_labels_DataSet2(model_name: str, base64_image: str):
    model_path = os.path.join("modelsAvailable/2", f"{model_name}.pt")
    yaml_path = "dataset.yaml"

    if not os.path.exists(model_path):
        return {"error": f"Modelo '{model_name}' não encontrado."}, 404

    class_map = load_class_names_from_yaml(yaml_path)

    # Decode do base64 para bytes
    try:
        image_bytes = base64.b64decode(base64_image)
    except Exception as e:
        return {"error": f"Imagem base64 inválida: {str(e)}"}, 400

    # Salvar temporariamente
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(image_bytes)
    temp.close()

    label_counts = {}
    image_bgr = cv2.imread(temp.name)

    if re.search(r"yolo", model_name, re.IGNORECASE):
        yolo_model = YOLO(model_path)
        results = yolo_model(temp.name)[0]

        class_ids = [int(cls) for cls in results.boxes.cls]
        label_names = [class_map.get(class_id, f"class_{class_id}") for class_id in class_ids]
        label_counts = dict(Counter(label_names))

        for box, cls in zip(results.boxes.xyxy, class_ids):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = class_map.get(cls, f"class_{cls}")
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elif re.search(r"rcnn|mobilenet|vgg", model_name, re.IGNORECASE):
        image = Image.open(temp.name).convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image)

        rcnn_model = torch.load(model_path, map_location="cpu")
        rcnn_model.eval()

        with torch.no_grad():
            outputs = rcnn_model([image_tensor])[0]

        min_score = 0.5
        for i in range(len(outputs["boxes"])):
            score = outputs["scores"][i].item()
            if score >= min_score:
                box = outputs["boxes"][i].tolist()
                class_id = int(outputs["labels"][i].item())
                label = class_map.get(class_id, f"class_{class_id}")
                label_counts[label] = label_counts.get(label, 0) + 1

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    else:
        os.remove(temp.name)
        return {"error": "Tipo de modelo não reconhecido no nome."}, 400

    image_base64 = encode_image_base64(image_bgr)
    os.remove(temp.name)

    return {
        "labels": list(label_counts.keys()),
        "counts": label_counts,
        "image_base64": image_base64
    }, 200

import os
import re
import cv2
import json
import base64
import tempfile
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from collections import Counter
from torchvision import models, transforms
from ultralytics import YOLO
import yaml

# Constantes de labels
SCENE_LABELS = ['city street', 'residential', 'highway', 'gas stations', 'parking', 'tunnel',
                'bridge', 'railroad', 'roundabout', 'construction', 'parking lot']
WEATHER_LABELS = ['clear', 'rainy', 'foggy', 'snowy', 'overcast', 'undefined', 'partly cloudy']
TIME_LABELS = ['daytime', 'night', 'dawn/dusk', 'undefined']

# Modelo de atributos
class AttributeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.scene_head = nn.Linear(num_features, len(SCENE_LABELS))
        self.weather_head = nn.Linear(num_features, len(WEATHER_LABELS))
        self.time_head = nn.Linear(num_features, len(TIME_LABELS))

    def forward(self, x):
        feat = self.backbone(x)
        return self.scene_head(feat), self.weather_head(feat), self.time_head(feat)

# Utilitário para carregar nomes de classes
def load_class_names_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return {i: name for i, name in enumerate(data["names"])}

# Utilitário para codificar imagem em base64
def encode_image_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def detect_labels_DataSet3(model_name: str, base64_image: str):
    model_path = os.path.join("modelsAvailable/3", "best_yolo.pt")
    attr_model_path = os.path.join("modelsAvailable/3", "attribute_classifier.pt")
    yaml_path = "dataset.yaml"

    if not os.path.exists(model_path) or not os.path.exists(attr_model_path):
        return {"error": "Modelo não encontrado."}, 404

    class_map = load_class_names_from_yaml(yaml_path)

    # Decode base64
    try:
        image_bytes = base64.b64decode(base64_image)
    except Exception as e:
        return {"error": f"Imagem base64 inválida: {str(e)}"}, 400

    # Salvar temporariamente
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(image_bytes)
    temp.close()

    label_counts = {}
    image_bgr = cv2.imread(temp.name)

    # ---------- Classificação de atributos ----------
    attr_model = AttributeClassifier()
    attr_model.load_state_dict(torch.load(attr_model_path, map_location="cpu"))
    attr_model.eval()

    transform_attr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_pil = Image.open(temp.name).convert("RGB")
    image_tensor = transform_attr(image_pil).unsqueeze(0)

    with torch.no_grad():
        scene_logits, weather_logits, time_logits = attr_model(image_tensor)
        scene_label = SCENE_LABELS[scene_logits.argmax().item()]
        weather_label = WEATHER_LABELS[weather_logits.argmax().item()]
        time_label = TIME_LABELS[time_logits.argmax().item()]

    # ---------- Detecção de objetos ----------
    if re.search(r"yolo", "yolo", re.IGNORECASE):
        yolo_model = YOLO(model_path)
        results = yolo_model(temp.name)[0]

        class_ids = [int(cls) for cls in results.boxes.cls]
        label_names = [class_map.get(class_id, f"class_{class_id}") for class_id in class_ids]
        label_counts = dict(Counter(label_names))

        for box, cls in zip(results.boxes.xyxy, class_ids):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = class_map.get(cls, f"class_{cls}")
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elif re.search(r"rcnn|mobilenet|vgg", model_name, re.IGNORECASE):
        image_tensor_rcnn = transforms.ToTensor()(image_pil)
        rcnn_model = torch.load(model_path, map_location="cpu")
        rcnn_model.eval()

        with torch.no_grad():
            outputs = rcnn_model([image_tensor_rcnn])[0]

        min_score = 0.5
        for i in range(len(outputs["boxes"])):
            score = outputs["scores"][i].item()
            if score >= min_score:
                box = outputs["boxes"][i].tolist()
                class_id = int(outputs["labels"][i].item())
                label = class_map.get(class_id, f"class_{class_id}")
                label_counts[label] = label_counts.get(label, 0) + 1

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    else:
        os.remove(temp.name)
        return {"error": "Tipo de modelo não reconhecido no nome."}, 400

    # Adiciona os atributos globais dentro das "labels"
    label_counts[scene_label] = 1
    label_counts[weather_label] = 1
    label_counts[time_label] = 1

    # Desenha os atributos na imagem (canto superior esquerdo)
    attr_text = f"Scene: {scene_label} | Weather: {weather_label} | Time: {time_label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_size, _ = cv2.getTextSize(attr_text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    margin_top = 20 + text_h

    # Verifica se há espaço para o texto
    if image_bgr.shape[0] < margin_top + 5:
        # Aumenta a altura da imagem para adicionar o texto no topo
        new_height = margin_top + image_bgr.shape[0]
        new_image = np.zeros((new_height, image_bgr.shape[1], 3), dtype=np.uint8)
        new_image[margin_top:, :, :] = image_bgr
        image_bgr = new_image

    # Desenha fundo preto e texto branco
    cv2.rectangle(image_bgr, (5, 5), (10 + text_w, 10 + text_h + 5), (0, 0, 0), -1)
    cv2.putText(image_bgr, attr_text, (10, 10 + text_h), font, font_scale, (255, 255, 255), font_thickness)

    image_base64 = encode_image_base64(image_bgr)
    os.remove(temp.name)

    return {
        "labels": list(label_counts.keys()),
        "counts": label_counts,
        "image_base64": image_base64
    }, 200


