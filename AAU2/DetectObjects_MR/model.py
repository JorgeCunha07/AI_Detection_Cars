import time
import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# ============================
# 1. Dataset with global attributes and support for poly2d boxes
# ============================
class CustomObjectDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Filter images that have a corresponding JSON file
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
            print("The following image files do not have corresponding JSON labels:")
            for m in missing:
                print(m)
        else:
            print("All images have corresponding JSON files.")

        # Load mapping files (assumed to be in the 'helpers' directory)
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
        image_path = os.path.join(self.images_dir, image_id + '.jpg')
        label_path = os.path.join(self.labels_dir, image_id + '.json')

        image = Image.open(image_path).convert("RGB")
        with open(label_path, 'r') as f:
            data = json.load(f)

        boxes = []
        labels = []
        # Loop over objects to get bounding boxes and class labels.
        for obj in data["frames"][0]["objects"]:
            category = obj["category"]
            if category in self.category_to_label:
                # Compute bounding box from box2d or poly2d
                if "box2d" in obj:
                    box = obj["box2d"]
                    bbox = [box["x1"], box["y1"], box["x2"], box["y2"]]
                elif "poly2d" in obj:
                    pts = obj["poly2d"]
                    xs = [pt[0] for pt in pts]
                    ys = [pt[1] for pt in pts]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    continue

                # Check if the bounding box has positive width and height
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if width <= 0 or height <= 0:
                    print(f"Invalid bounding box found for image {image_id}. Skipping.")
                    continue

                boxes.append(bbox)
                labels.append(self.category_to_label[category])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # If no valid boxes were found, add a dummy box to avoid errors downstream
        if len(boxes) == 0:
            print(f"No valid boxes found for image {image_id}. Adding a dummy box.")
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        # Process global attributes
        attributes = data["attributes"]
        weather_label = self.weather_to_label[attributes["weather"]]
        scene_label = self.scene_to_label[attributes["scene"]]
        timeofday_label = self.timeofday_to_label[attributes["timeofday"]]
        global_attrs = {
            "weather": torch.tensor(weather_label, dtype=torch.long),
            "scene": torch.tensor(scene_label, dtype=torch.long),
            "timeofday": torch.tensor(timeofday_label, dtype=torch.long)
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

# ============================
# Main function with training, evaluation, and saving model
# ============================
def main():
    # Define transformation and datasets
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CustomObjectDetectionDataset(
        images_dir="images/train",
        labels_dir="labels/train",
        transform=transform
    )
    val_dataset = CustomObjectDetectionDataset(
        images_dir="images/val",
        labels_dir="labels/val",
        transform=transform
    )

    # Use only 5% of training data for quick testing
    train_subset_size = max(1, int(len(train_dataset) * 0.005))
    train_indices = random.sample(range(len(train_dataset)), train_subset_size)
    train_subset = Subset(train_dataset, train_indices)
    print(f"Using {train_subset_size} out of {len(train_dataset)} training images (5%).")

    # DataLoader with increased batch size, pin_memory, prefetch_factor and num_workers
    train_loader = DataLoader(
        train_subset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # ============================
    # Detection model (Faster R-CNN) and Backbone
    # ============================
    mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    backbone = mobilenet.features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2
    )

    detection_model = FasterRCNN(
        backbone,
        num_classes=len(train_dataset.category_to_label) + 1,  # background + detected categories
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    detection_model = detection_model.to(device)

    # Determine attribute class counts from mapping files:
    weather_classes = len(train_dataset.weather_to_label)
    scene_classes = len(train_dataset.scene_to_label)
    timeofday_classes = len(train_dataset.timeofday_to_label)
    print(f"Weather classes: {weather_classes}, Scene classes: {scene_classes}, Timeofday classes: {timeofday_classes}")

    # ============================
    # Multi-task Model: Detection + Global Attributes
    # ============================
    class MultiTaskModel(torch.nn.Module):
        def __init__(self, detection_model, backbone, in_features=1280,
                     weather_classes=weather_classes,
                     scene_classes=scene_classes,
                     timeofday_classes=timeofday_classes):
            super(MultiTaskModel, self).__init__()
            self.detection_model = detection_model
            self.backbone = backbone  # Reuse same backbone
            self.attr_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc_weather = torch.nn.Linear(in_features, weather_classes)
            self.fc_scene = torch.nn.Linear(in_features, scene_classes)
            self.fc_timeofday = torch.nn.Linear(in_features, timeofday_classes)

        def forward(self, images, targets=None, global_attrs=None):
            # Detection part
            if self.training:
                detection_loss = self.detection_model(images, targets)
            else:
                detection_loss = {}
                detections = self.detection_model(images)

            # Global attributes part
            imgs_tensor = torch.stack(images)  # [N, C, H, W]
            feats = self.backbone(imgs_tensor)   # [N, 1280, H', W']
            pooled = self.attr_pool(feats)         # [N, 1280, 1, 1]
            pooled = pooled.view(pooled.size(0), -1) # [N, 1280]
            weather_logits = self.fc_weather(pooled)   # [N, weather_classes]
            scene_logits = self.fc_scene(pooled)         # [N, scene_classes]
            timeofday_logits = self.fc_timeofday(pooled) # [N, timeofday_classes]

            # Calculate attribute loss if training
            if self.training and global_attrs is not None:
                weather_labels = torch.stack([attr["weather"] for attr in global_attrs]).to(images[0].device)
                scene_labels = torch.stack([attr["scene"] for attr in global_attrs]).to(images[0].device)
                timeofday_labels = torch.stack([attr["timeofday"] for attr in global_attrs]).to(images[0].device)
                loss_weather = torch.nn.functional.cross_entropy(weather_logits, weather_labels)
                loss_scene = torch.nn.functional.cross_entropy(scene_logits, scene_labels)
                loss_timeofday = torch.nn.functional.cross_entropy(timeofday_logits, timeofday_labels)
                attr_loss = loss_weather + loss_scene + loss_timeofday
            else:
                attr_loss = 0

            return detection_loss, attr_loss, weather_logits, scene_logits, timeofday_logits

    multi_task_model = MultiTaskModel(detection_model, backbone).to(device)

    # ============================
    # Training with AMP, batch progress, time, and batch size measurements
    # ============================
    optimizer = torch.optim.SGD(multi_task_model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = 5
    for epoch in range(num_epochs):
        multi_task_model.train()
        epoch_start = time.time()
        epoch_batch_sizes = []  # List to store batch sizes for the epoch
        total_batches = len(train_loader)
        for batch_idx, (images, targets, global_attrs) in enumerate(train_loader, start=1):
            batch_start = time.time()
            current_batch_size = len(images)
            epoch_batch_sizes.append(current_batch_size)

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            # Use AMP autocast for mixed precision training
            with torch.cuda.amp.autocast():
                detection_loss, attr_loss, _, _, _ = multi_task_model(images, targets, global_attrs)
                total_detection_loss = sum(loss for loss in detection_loss.values())
                total_loss = total_detection_loss + attr_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_time = time.time() - batch_start
            image_time = batch_time / current_batch_size

            print(f"Epoch {epoch+1}, Batch {batch_idx}/{total_batches}: {current_batch_size} images processed in {batch_time:.4f} s (avg {image_time:.4f} s/image)")
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} s")
        print(f"Batch sizes for epoch {epoch+1}: {epoch_batch_sizes}")

    # ============================
    # Evaluation of Detection and Attributes
    # ============================
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def evaluate_model(multi_task_model, dataloader, iou_threshold=0.5, confidence_threshold=0.6):
        multi_task_model.eval()
        total_gt = 0   # Total ground truth objects
        total_tp = 0   # Total true positives in detection
        correct_weather = 0
        correct_scene = 0
        correct_timeofday = 0
        total_images = 0

        with torch.no_grad():
            for images, targets, global_attrs in dataloader:
                images = [img.to(device) for img in images]

                # Evaluate detection using the internal detection_model
                detections = multi_task_model.detection_model(images)
                for i in range(len(detections)):
                    gt_boxes = targets[i]["boxes"].cpu().numpy()
                    gt_labels = targets[i]["labels"].cpu().numpy()
                    total_gt += len(gt_boxes)

                    pred_boxes = detections[i]["boxes"].cpu().numpy()
                    pred_labels = detections[i]["labels"].cpu().numpy()
                    pred_scores = detections[i]["scores"].cpu().numpy()

                    keep = pred_scores >= confidence_threshold
                    pred_boxes = pred_boxes[keep]
                    pred_labels = pred_labels[keep]

                    matched_pred = set()
                    for j, gt_box in enumerate(gt_boxes):
                        gt_label = gt_labels[j]
                        best_iou = 0
                        best_idx = -1
                        for k, pred_box in enumerate(pred_boxes):
                            if k in matched_pred:
                                continue
                            if pred_labels[k] != gt_label:
                                continue
                            iou = compute_iou(gt_box, pred_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = k
                        if best_iou >= iou_threshold:
                            total_tp += 1
                            matched_pred.add(best_idx)

                # Evaluate global attributes:
                images_tensor = torch.stack(images)
                feats = multi_task_model.backbone(images_tensor)
                pooled = multi_task_model.attr_pool(feats)
                pooled = pooled.view(pooled.size(0), -1)
                weather_logits = multi_task_model.fc_weather(pooled)
                scene_logits = multi_task_model.fc_scene(pooled)
                timeofday_logits = multi_task_model.fc_timeofday(pooled)

                weather_preds = torch.argmax(weather_logits, dim=1)
                scene_preds = torch.argmax(scene_logits, dim=1)
                timeofday_preds = torch.argmax(timeofday_logits, dim=1)

                for i, attrs in enumerate(global_attrs):
                    correct_weather += (weather_preds[i].cpu() == attrs["weather"].cpu()).item()
                    correct_scene += (scene_preds[i].cpu() == attrs["scene"].cpu()).item()
                    correct_timeofday += (timeofday_preds[i].cpu() == attrs["timeofday"].cpu()).item()
                    total_images += 1

        detection_recall = total_tp / total_gt if total_gt > 0 else 0
        weather_acc = correct_weather / total_images if total_images > 0 else 0
        scene_acc = correct_scene / total_images if total_images > 0 else 0
        timeofday_acc = correct_timeofday / total_images if total_images > 0 else 0

        return detection_recall, weather_acc, scene_acc, timeofday_acc

    detection_recall, weather_acc, scene_acc, timeofday_acc = evaluate_model(multi_task_model, val_loader, iou_threshold=0.5, confidence_threshold=0.6)
    print(f"Detection recall on the validation set: {detection_recall * 100:.2f}%")
    print(f"Weather accuracy: {weather_acc * 100:.2f}%")
    print(f"Scene accuracy: {scene_acc * 100:.2f}%")
    print(f"Timeofday accuracy: {timeofday_acc * 100:.2f}%")

    # ============================
    # Save the model for future inference and visualization
    # ============================
    torch.save(multi_task_model.state_dict(), "multi_task_model.pth")
    print("Model saved as 'multi_task_model.pth'.")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Only needed if freezing the app to an executable.
    main()
