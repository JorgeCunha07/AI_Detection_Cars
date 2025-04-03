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
        return {"error": "Tipo de modelo não reconhecido no nome."}, 400
