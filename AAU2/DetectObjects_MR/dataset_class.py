import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


###########################################
# 1. Dataset e Collate Function
###########################################
class MultiTaskObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Filtra as imagens que possuem JSON correspondente
        all_ids = [f.split('.')[0] for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.image_ids = []
        missing = []
        for img_id in all_ids:
            json_path = os.path.join(labels_dir, img_id + '.json')
            if os.path.exists(json_path):
                self.image_ids.append(img_id)
            else:
                missing.append(img_id)
        if missing:
            print("Os seguintes arquivos de imagem não possuem JSON:")
            for m in missing:
                print(m)
        else:
            print("Todas as imagens possuem JSON correspondente.")

        # Carrega os mapeamentos (assumindo que estão na pasta 'helpers')
        with open('helpers/categories.json', 'r') as f:
            self.category_to_label = json.load(f)
        with open('helpers/weather.json', 'r') as f:
            self.weather_to_label = json.load(f)
        with open('helpers/scene.json', 'r') as f:
            self.scene_to_label = json.load(f)
        with open('helpers/timeofday.json', 'r') as f:
            self.timeofday_to_label = json.load(f)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        try:
            image = Image.open(os.path.join(self.images_dir, image_id + ".jpg")).convert("RGB")
            with open(os.path.join(self.labels_dir, image_id + ".json")) as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Erro ao carregar {image_id}: {e}")
            return None

        boxes, labels = [], []
        for obj in data["frames"][0]["objects"]:
            category = obj.get("category")
            if category not in self.category_to_label:
                continue
            bbox = None
            if "box2d" in obj:
                b = obj["box2d"]
                bbox = [b["x1"], b["y1"], b["x2"], b["y2"]]
            elif "poly2d" in obj:
                pts_raw = obj["poly2d"]
                if isinstance(pts_raw[0], dict) and "vertices" in pts_raw[0]:
                    pts = pts_raw[0]["vertices"]
                else:
                    pts = [(p[0], p[1]) for p in pts_raw if isinstance(p, (list, tuple)) and len(p) >= 2]
                if len(pts) >= 2:
                    xs, ys = zip(*pts)
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
            if bbox and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                boxes.append(bbox)
                labels.append(self.category_to_label[category])

        if not boxes:
            return None

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        attrs = data["attributes"]
        global_attrs = {
            "weather": torch.tensor(self.weather_to_label.get(attrs["weather"], 0)),
            "scene": torch.tensor(self.scene_to_label.get(attrs["scene"], 0)),
            "timeofday": torch.tensor(self.timeofday_to_label.get(attrs["timeofday"], 0)),
        }

        if self.transform:
            image = self.transform(image)

        return image, target, global_attrs


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return ([], [], [])
    images, targets, global_attrs = zip(*batch)
    return list(images), list(targets), list(global_attrs)