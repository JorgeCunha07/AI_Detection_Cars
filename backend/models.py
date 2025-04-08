# models.py
import json
import sys
import tempfile
import os
import re
import torch
import torch.nn as nn
import yaml
import base64
import tempfile
import cv2
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from collections import Counter
from torchvision import models, transforms
from ultralytics import YOLO
import base64
import numpy as np

from multitaskmodel import MultiTaskModel

# Insere a classe MultiTaskModel no namespace __main__
sys.modules['__main__'].MultiTaskModel = MultiTaskModel


def load_class_names_from_yaml(yaml_path="dataset.yaml"):
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return {i: name for i, name in enumerate(data.get("names", []))}

def encode_image_base64_1(image_array):
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode("utf-8")

def encode_image_base64_2(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

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

    image_base64 = encode_image_base64_1(image_bgr)
    os.remove(temp.name)

    return {
        "labels": list(label_counts.keys()),
        "counts": label_counts,
        "image_base64": image_base64
    }, 200


def detect_labels_DataSet2(model_name: str, base64_image: str):
    # Monta o caminho para o modelo multi-tarefa (no exemplo, espera-se um arquivo .pth)
    model_path = os.path.join("modelsAvailable/2", f"{model_name}.pth")
    if not os.path.exists(model_path):
        return {"error": f"Modelo '{model_name}' não encontrado."}, 404

    # Carrega os mapeamentos para categorias e atributos globais
    with open("helpers/categories.json", "r") as f:
        category_to_label = json.load(f)
    with open("helpers/weather.json", "r") as f:
        weather_to_label = json.load(f)
    with open("helpers/scene.json", "r") as f:
        scene_to_label = json.load(f)
    with open("helpers/timeofday.json", "r") as f:
        timeofday_to_label = json.load(f)

    # Define a transformação da imagem
    transform = transforms.Compose([transforms.ToTensor()])

    # Carrega o modelo multi-tarefa
    multi_task_model = torch.load(model_path, map_location="cpu")
    multi_task_model.eval()

    # Decodifica a imagem base64 e salva em um arquivo temporário
    try:
        image_bytes = base64.b64decode(base64_image)
    except Exception as e:
        return {"error": f"Imagem base64 inválida: {str(e)}"}, 400

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.write(image_bytes)
    temp_file.close()

    # Função interna para realizar a inferência e anotar a imagem
    def infer_image(image_path):
        # Inverte os mapeamentos para poder converter rótulos numéricos para nomes
        inv_category = {v: k for k, v in category_to_label.items()}
        inv_weather = {v: k for k, v in weather_to_label.items()}
        inv_scene = {v: k for k, v in scene_to_label.items()}
        inv_timeofday = {v: k for k, v in timeofday_to_label.items()}

        # Abre a imagem original e converte para RGB
        orig_image = Image.open(image_path).convert("RGB")
        input_image = transform(orig_image).to(torch.device("cpu")).unsqueeze(0)

        with torch.no_grad():
            # Inferência da parte de detecção
            detections = multi_task_model.detection_model([input_image.squeeze(0)])
            # Inferência dos atributos globais
            feats = multi_task_model.backbone(input_image)
            pooled = multi_task_model.attr_pool(feats)
            pooled = pooled.view(pooled.size(0), -1)
            weather_logits = multi_task_model.fc_weather(pooled)
            scene_logits = multi_task_model.fc_scene(pooled)
            timeofday_logits = multi_task_model.fc_timeofday(pooled)
            weather_pred = int(torch.argmax(weather_logits, dim=1).item())
            scene_pred = int(torch.argmax(scene_logits, dim=1).item())
            timeofday_pred = int(torch.argmax(timeofday_logits, dim=1).item())
            global_attributes = {
                "weather": inv_weather.get(weather_pred, str(weather_pred)),
                "scene": inv_scene.get(scene_pred, str(scene_pred)),
                "timeofday": inv_timeofday.get(timeofday_pred, str(timeofday_pred))
            }

            detection = detections[0]
            boxes = detection["boxes"].cpu().numpy().tolist()
            labels_detection = detection["labels"].cpu().numpy().tolist()
            scores = detection["scores"].cpu().numpy().tolist()

            detection_list = []
            # Filtra as detecções abaixo do limiar de confiança e monta a lista
            for bbox, label, score in zip(boxes, labels_detection, scores):
                if score < 0.5:
                    continue
                detection_list.append({
                    "category": inv_category.get(label, str(label)),
                    "score": score,
                    "box": bbox
                })

        # Anota a imagem com as caixas delimitadoras e rótulos
        draw = ImageDraw.Draw(orig_image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            font = ImageFont.load_default()

        for det in detection_list:
            bbox = det["box"]
            text = f"{det['category']}: {det['score']:.2f}"
            draw.rectangle(bbox, outline="red", width=2)
            draw.text((bbox[0], bbox[1] - 10), text, fill="red", font=font)

        # Exibe os atributos globais na imagem
        attr_text = (f"Weather: {global_attributes['weather']}, "
                     f"Scene: {global_attributes['scene']}, "
                     f"Time: {global_attributes['timeofday']}")
        draw.text((10, 10), attr_text, fill="blue", font=font)

        # Constrói os outputs de rótulos e contagens a partir das detecções
        labels_found = [det["category"] for det in detection_list]
        label_counts = dict(Counter(labels_found))
        unique_labels = list(label_counts.keys())

        return orig_image, unique_labels, label_counts

    # Executa a inferência na imagem temporária
    annotated_image, labels, counts = infer_image(temp_file.name)
    os.remove(temp_file.name)

    image_base64_annotated = encode_image_base64_2(annotated_image)

    return {
        "labels": labels,
        "counts": counts,
        "image_base64": image_base64_annotated
    }, 200



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

def detect_labels_DataSet3(model_name: str, base64_image: str):
    model_path = os.path.join("modelsAvailable/3", f"best_yolo.pt")
    attr_model_path = os.path.join("modelsAvailable/3", f"attribute_classifier.pt")
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

    image_base64 = encode_image_base64_1(image_bgr)
    os.remove(temp.name)

    return {
        "labels": list(label_counts.keys()),
        "counts": label_counts,
        "image_base64": image_base64
    }, 200