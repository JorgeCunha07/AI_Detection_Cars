# models.py
import base64
import base64
import base64
import cv2
import json
import os
import re
import sys
import tempfile
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from io import BytesIO
from torchvision import transforms
from ultralytics import YOLO

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


def detect_labels_DataSet3(model_name: str, base64_image: str):
    model_path = os.path.join("modelsAvailable/3", f"{model_name}.pt")
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
