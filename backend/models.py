import base64
import cv2
import json
import numpy as np
import os
import re
import sys
import tempfile
import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from io import BytesIO
from torchvision import models, transforms
from ultralytics import YOLO

from multitaskmodel import MultiTaskModel
sys.modules['__main__'].MultiTaskModel = MultiTaskModel

###########################
# Funções Utilitárias (Reaproveitáveis)
###########################
class BaseDetector:
    @staticmethod
    def decode_image(base64_image: str) -> (Image.Image, str):
        """Decodifica a imagem base64 e retorna o objeto PIL e o caminho temporário."""
        try:
            image_bytes = base64.b64decode(base64_image)
        except Exception as e:
            raise ValueError(f"Imagem base64 inválida: {str(e)}")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(image_bytes)
        temp_file.close()
        pil_image = Image.open(temp_file.name).convert("RGB")
        return pil_image, temp_file.name

    @staticmethod
    def cleanup_temp(file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def encode_image_pil_to_base64(pil_image: Image.Image) -> str:
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def get_text_size(font: ImageFont.FreeTypeFont, text: str):
        """
        Tenta usar font.getsize(), e se não estiver disponível usa font.getbbox().
        Retorna uma tupla (width, height).
        """
        try:
            return font.getsize(text)
        except AttributeError:
            bbox = font.getbbox(text)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    @staticmethod
    def draw_text_with_background(draw: ImageDraw.Draw, xy: tuple, text: str, font: ImageFont.FreeTypeFont,
                                  text_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """
        Desenha um texto com fundo sólido para aumentar a legibilidade.
        :param draw: objeto ImageDraw.
        :param xy: posição (x, y) para desenhar o texto.
        :param text: o texto a ser desenhado.
        :param font: fonte PIL utilizada para o texto.
        :param text_color: cor do texto (RGB).
        :param bg_color: cor de fundo (RGB).
        """
        x, y = xy
        text_w, text_h = BaseDetector.get_text_size(font, text)
        # Adiciona uma margem de 4 pixels
        draw.rectangle([x, y, x + text_w + 4, y + text_h + 4], fill=bg_color)
        draw.text((x + 2, y + 2), text, font=font, fill=text_color)

    @staticmethod
    def draw_detections_with_bg(pil_image: Image.Image, detections: list, color: tuple, font: ImageFont.FreeTypeFont):
        """
        Desenha os bounding boxes e os textos das detecções com fundo para o texto.
        :param pil_image: imagem PIL onde os resultados serão desenhados.
        :param detections: lista de detecções, cada uma um dicionário com {"category", "score", "box"}.
        :param color: cor para o contorno e fundo (RGB), ex.: (0, 255, 0).
        :param font: fonte PIL utilizada para o texto.
        """
        draw = ImageDraw.Draw(pil_image)
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            text = f"{det['category']}:{det['score']:.2f}"
            # Desenha a bounding box com uma largura maior para destacar
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Calcula a posição do texto: tenta acima do box, mas se não couber, desenha dentro
            text_x = x1
            text_y = y1 - 24 if (y1 - 24) > 0 else y1
            BaseDetector.draw_text_with_background(draw, (text_x, text_y), text, font, text_color=(255, 255, 255), bg_color=(0, 0, 0))

def invert_mapping(mapping: dict) -> dict:
    """Cria um dicionário inverso a partir do mapping fornecido.
       Se o valor puder ser convertido para int, usa o int como chave."""
    inv = {}
    for k, v in mapping.items():
        try:
            inv[int(v)] = k
        except:
            inv[v] = k
    return inv

###########################
# Detector para DataSet1 (usando YOLO e YAML)
###########################
class DetectorDataSet1(BaseDetector):
    @staticmethod
    def load_class_map(yaml_path="dataset.yaml"):
        if not os.path.exists(yaml_path):
            return {}
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return {i: name for i, name in enumerate(data.get("names", []))}

    @staticmethod
    def get_detections(model_name: str, base64_image: str):
        """
        Executa o modelo do DataSet1.
        Se o nome do modelo contiver "yolo", utiliza o modelo YOLO;
        se contiver "rcnn", "mobilenet" ou "vgg", utiliza a arquitetura RCNN/MobileNet.
        Retorna:
          - status: 200 ou dicionário de erro.
          - detection_list: lista de detecções com {"category", "score", "box"}.
          - label_counts: dicionário com a contagem de cada rótulo.
        """
        model_path = os.path.join("modelsAvailable", "1", f"{model_name}.pt")
        if not os.path.exists(model_path):
            return {"error": f"Modelo '{model_name}' não encontrado em DataSet1."}, 404, [], {}
        class_map = DetectorDataSet1.load_class_map("dataset.yaml")
        try:
            _, temp_path = BaseDetector.decode_image(base64_image)
        except ValueError as e:
            return {"error": str(e)}, 400, [], {}

        detection_list = {}
        label_counts = {}
        detection_list = []
        # Se o modelo é YOLO
        if re.search(r"yolo", model_name, re.IGNORECASE):
            yolo_model = YOLO(model_path)
            results = yolo_model(temp_path)[0]
            class_ids = [int(cls) for cls in results.boxes.cls]
            for box, cls, conf in zip(results.boxes.xyxy, class_ids, results.boxes.conf):
                score = float(conf)
                if score < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                label = class_map.get(cls, f"class_{cls}")
                detection_list.append({
                    "category": label,
                    "score": score,
                    "box": [x1, y1, x2, y2]
                })
                label_counts[label] = label_counts.get(label, 0) + 1
        # Caso o modelo seja Faster RCNN / MobileNet / VGG
        elif re.search(r"rcnn|mobilenet|vgg", model_name, re.IGNORECASE):
            # Usa o PIL para carregar a imagem
            image = Image.open(temp_path).convert("RGB")
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
                    detection_list.append({
                        "category": label,
                        "score": score,
                        "box": list(map(int, box))
                    })
                    label_counts[label] = label_counts.get(label, 0) + 1
        else:
            BaseDetector.cleanup_temp(temp_path)
            return {"error": "Tipo de modelo não reconhecido no nome."}, 400, [], {}

        BaseDetector.cleanup_temp(temp_path)
        return 200, detection_list, label_counts

###########################
# Detector para DataSet2 (usando modelo multi-tarefa e JSON)
###########################
class DetectorDataSet2(BaseDetector):
    @staticmethod
    def load_mappings():
        with open("helpers/categories.json", "r") as f:
            cat_map = json.load(f)
        with open("helpers/weather.json", "r") as f:
            weather_map = json.load(f)
        with open("helpers/scene.json", "r") as f:
            scene_map = json.load(f)
        with open("helpers/timeofday.json", "r") as f:
            time_map = json.load(f)
        return cat_map, weather_map, scene_map, time_map

    @staticmethod
    def get_detections(model_name: str, base64_image: str):
        """
        Executa o modelo multi-tarefa do DataSet2.
        Retorna:
          - status: 200 ou dicionário de erro.
          - detection_list: lista de detecções com {"category", "score", "box"}.
          - label_counts: dicionário com as contagens dos rótulos de detecção.
          - global_attributes: dicionário com {"weather", "scene", "timeofday"}.
        """
        model_path = os.path.join("modelsAvailable", "2", f"{model_name}.pth")
        if not os.path.exists(model_path):
            return {"error": f"Modelo '{model_name}' não encontrado em DataSet2."}, 404, [], {}, {}
        cat_map, weather_map, scene_map, time_map = DetectorDataSet2.load_mappings()
        # Cria os mapeamentos inversos para converter os valores numéricos retornados pelo modelo em rótulos
        inv_cat = invert_mapping(cat_map)
        inv_weather = invert_mapping(weather_map)
        inv_scene = invert_mapping(scene_map)
        inv_time = invert_mapping(time_map)

        transform = transforms.Compose([transforms.ToTensor()])
        multi_task_model = torch.load(model_path, map_location="cpu")
        multi_task_model.eval()
        try:
            pil_image, temp_path = BaseDetector.decode_image(base64_image)
        except ValueError as e:
            return {"error": str(e)}, 400, [], {}, {}
        input_image = transform(pil_image).unsqueeze(0)
        detection_list = []
        with torch.no_grad():
            detections = multi_task_model.detection_model([input_image.squeeze(0)])
            feats = multi_task_model.backbone(input_image)
            pooled = multi_task_model.attr_pool(feats).view(input_image.size(0), -1)
            weather_logits = multi_task_model.fc_weather(pooled)
            scene_logits = multi_task_model.fc_scene(pooled)
            time_logits = multi_task_model.fc_timeofday(pooled)
            weather_pred = int(torch.argmax(weather_logits, dim=1).item())
            scene_pred = int(torch.argmax(scene_logits, dim=1).item())
            timeofday_pred = int(torch.argmax(time_logits, dim=1).item())
            global_attributes = {
                "weather": inv_weather.get(weather_pred, str(weather_pred)),
                "scene": inv_scene.get(scene_pred, str(scene_pred)),
                "timeofday": inv_time.get(timeofday_pred, str(timeofday_pred))
            }
            det = detections[0]
            boxes = det["boxes"].cpu().numpy().tolist()
            labels_det = det["labels"].cpu().numpy().tolist()
            scores = det["scores"].cpu().numpy().tolist()
            for bbox, lbl, sc in zip(boxes, labels_det, scores):
                if sc < 0.5:
                    continue
                cat_label = inv_cat.get(lbl, f"class_{lbl}")
                detection_list.append({
                    "category": cat_label,
                    "score": sc,
                    "box": bbox
                })
        detection_labels = [d["category"] for d in detection_list]
        label_counts = dict(Counter(detection_labels))
        for attr in [global_attributes["weather"], global_attributes["scene"], global_attributes["timeofday"]]:
            label_counts[attr] = label_counts.get(attr, 0) + 1
        BaseDetector.cleanup_temp(temp_path)
        return 200, detection_list, label_counts, global_attributes

###########################
# Funções para o Frontend (mantidas)
###########################
def detect_labels_DataSet1(model_name: str, base64_image: str):
    """Função para o frontend – usa DetectorDataSet1 e desenha as detecções na imagem."""
    status, detection_list, label_counts = DetectorDataSet1.get_detections(model_name, base64_image)
    if status != 200:
        return {"error": "Erro no DataSet1"}, status
    try:
        pil_image, temp_path = BaseDetector.decode_image(base64_image)
    except ValueError as e:
        return {"error": str(e)}, 400
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    BaseDetector.draw_detections_with_bg(pil_image, detection_list, (0, 255, 0), font)
    image_base64 = BaseDetector.encode_image_pil_to_base64(pil_image)
    return {
        "labels": list(label_counts.keys()),
        "counts": label_counts,
        "image_base64": image_base64
    }, 200

def detect_labels_DataSet2(model_name: str, base64_image: str):
    """Função para o frontend – usa DetectorDataSet2 e desenha as detecções e os atributos globais na imagem."""
    status, detection_list, label_counts, global_attributes = DetectorDataSet2.get_detections(model_name, base64_image)
    if status != 200:
        return {"error": "Erro no DataSet2"}, status
    try:
        pil_image, temp_path = BaseDetector.decode_image(base64_image)
    except ValueError as e:
        return {"error": str(e)}, 400
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    BaseDetector.draw_detections_with_bg(pil_image, detection_list, (0, 0, 255), font)
    draw = ImageDraw.Draw(pil_image)
    attr_text = f"Weather: {global_attributes['weather']}, Scene: {global_attributes['scene']}, Time: {global_attributes['timeofday']}"
    BaseDetector.draw_text_with_background(draw, (10, 10), attr_text, font)
    image_base64 = BaseDetector.encode_image_pil_to_base64(pil_image)
    combined_labels = list(set([d["category"] for d in detection_list] + list(global_attributes.values())))
    return {
        "labels": combined_labels,
        "counts": label_counts,
        "image_base64": image_base64
    }, 200

###########################
# Função Combinada para DataSet3
###########################
def detect_labels_DataSet3(base64_image: str):
    """
    Combina os resultados dos modelos de DataSet1 e DataSet2 em uma única imagem:
      - Usa DetectorDataSet1 e DetectorDataSet2 para obter as detecções e atributos.
      - Desenha os bounding boxes (verde para DataSet1 e azul para DataSet2) e os textos com fundo.
      - Escreve os atributos globais no canto superior.
      - Retorna a união dos rótulos e as contagens combinadas.
    """
    try:
        decoded_data = base64.b64decode(base64_image)
    except Exception as e:
        return {"error": f"Imagem base64 inválida: {str(e)}"}, 400
    pil_image = Image.open(BytesIO(decoded_data)).convert("RGB")
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    status1, det_list1, counts1 = DetectorDataSet1.get_detections("best_yolo", base64_image)
    if status1 != 200:
        return {"error": "Erro no DataSet1"}, status1
    status2, det_list2, counts2, attrs2 = DetectorDataSet2.get_detections("vgg", base64_image)
    if status2 != 200:
        return {"error": "Erro no DataSet2"}, status2
    BaseDetector.draw_detections_with_bg(pil_image, det_list1, (0, 255, 0), font)
    BaseDetector.draw_detections_with_bg(pil_image, det_list2, (0, 0, 255), font)
    draw = ImageDraw.Draw(pil_image)
    attr_text = f"Weather: {attrs2['weather']}, Scene: {attrs2['scene']}, Time: {attrs2['timeofday']}"
    BaseDetector.draw_text_with_background(draw, (10, 10), attr_text, font)
    combined_counts = {}
    for k, v in counts1.items():
        combined_counts[k] = v
    for k, v in counts2.items():
        combined_counts[k] = combined_counts.get(k, 0) + v
    labels_ds1 = list({d["category"] for d in det_list1})
    labels_ds2 = list({d["category"] for d in det_list2})
    attrs_labels = list(attrs2.values())
    combined_labels = list(set(labels_ds1 + labels_ds2 + attrs_labels))
    final_image_base64 = BaseDetector.encode_image_pil_to_base64(pil_image)
    return {
        "labels": combined_labels,
        "counts": combined_counts,
        "image_base64": final_image_base64
    }, 200
