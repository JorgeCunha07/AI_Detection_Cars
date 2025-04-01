import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# Helper function to invert mapping dictionaries
def invert_mapping(mapping):
    return {v: k for k, v in mapping.items()}


# Define the inference function
def infer_and_visualize(image_path, model, device, transform,
                        category_to_label, weather_to_label,
                        scene_to_label, timeofday_to_label,
                        output_image_path="inference_output.jpg",
                        output_json_path="inference_output.json"):
    # Create inverse mappings for converting numerical predictions back to strings
    inv_category = invert_mapping(category_to_label)
    inv_weather = invert_mapping(weather_to_label)
    inv_scene = invert_mapping(scene_to_label)
    inv_timeofday = invert_mapping(timeofday_to_label)

    # Load and preprocess the image
    orig_image = Image.open(image_path).convert("RGB")
    input_image = transform(orig_image).to(device)
    input_image = input_image.unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        # Get detections from the detection branch
        detections = model.detection_model([input_image.squeeze(0)])

        # Get global attribute predictions using the backbone and attribute FC layers
        feats = model.backbone(input_image)
        pooled = model.attr_pool(feats)
        pooled = pooled.view(pooled.size(0), -1)
        weather_logits = model.fc_weather(pooled)
        scene_logits = model.fc_scene(pooled)
        timeofday_logits = model.fc_timeofday(pooled)

        # Compute predicted classes (using argmax)
        weather_pred = int(torch.argmax(weather_logits, dim=1).item())
        scene_pred = int(torch.argmax(scene_logits, dim=1).item())
        timeofday_pred = int(torch.argmax(timeofday_logits, dim=1).item())

        global_attributes = {
            "weather": inv_weather.get(weather_pred, str(weather_pred)),
            "scene": inv_scene.get(scene_pred, str(scene_pred)),
            "timeofday": inv_timeofday.get(timeofday_pred, str(timeofday_pred))
        }

        # Process detections for the single image
        detection = detections[0]
        boxes = detection["boxes"].cpu().numpy().tolist()
        labels = detection["labels"].cpu().numpy().tolist()
        scores = detection["scores"].cpu().numpy().tolist()

        # Filter detections with a low confidence (threshold = 0.5 here)
        detection_list = []
        for bbox, label_idx, score in zip(boxes, labels, scores):
            if score < 0.5:
                continue
            detection_list.append({
                "category": inv_category.get(label_idx, str(label_idx)),
                "score": score,
                "box": bbox
            })

    # Draw the detections on the image
    draw = ImageDraw.Draw(orig_image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw each detection box and its label
    for det in detection_list:
        bbox = det["box"]
        category_str = det["category"]
        score_str = f"{det['score']:.2f}"
        draw.rectangle(bbox, outline="red", width=2)
        text = f"{category_str}: {score_str}"
        draw.text((bbox[0], bbox[1] - 10), text, fill="red", font=font)

    # Draw global attribute text on the image
    attr_text = f"Weather: {global_attributes['weather']}, Scene: {global_attributes['scene']}, Time: {global_attributes['timeofday']}"
    draw.text((10, 10), attr_text, fill="blue", font=font)

    # Save the annotated image
    orig_image.save(output_image_path)
    print(f"Annotated image saved to {output_image_path}")

    # Save output JSON
    output = {
        "global_attributes": global_attributes,
        "detections": detection_list
    }
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output JSON saved to {output_json_path}")

    return output


# Define MultiTaskModel class exactly as in training
class MultiTaskModel(torch.nn.Module):
    def __init__(self, detection_model, backbone, in_features=1280,
                 weather_classes=None, scene_classes=None, timeofday_classes=None):
        super(MultiTaskModel, self).__init__()
        self.detection_model = detection_model
        self.backbone = backbone
        self.attr_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc_weather = torch.nn.Linear(in_features, weather_classes)
        self.fc_scene = torch.nn.Linear(in_features, scene_classes)
        self.fc_timeofday = torch.nn.Linear(in_features, timeofday_classes)

    def forward(self, images, targets=None, global_attrs=None):
        if self.training:
            detection_loss = self.detection_model(images, targets)
        else:
            detections = self.detection_model(images)
            detection_loss = detections  # In eval mode, we use detections.
        imgs_tensor = torch.stack(images)
        feats = self.backbone(imgs_tensor)
        pooled = self.attr_pool(feats)
        pooled = pooled.view(pooled.size(0), -1)
        weather_logits = self.fc_weather(pooled)
        scene_logits = self.fc_scene(pooled)
        timeofday_logits = self.fc_timeofday(pooled)
        return detection_loss, 0, weather_logits, scene_logits, timeofday_logits


if __name__ == '__main__':
    # Device and transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    # Load mapping files (update paths as needed)
    with open('helpers/categories.json', 'r') as f:
        category_to_label = json.load(f)
    with open('helpers/weather.json', 'r') as f:
        weather_to_label = json.load(f)
    with open('helpers/scene.json', 'r') as f:
        scene_to_label = json.load(f)
    with open('helpers/timeofday.json', 'r') as f:
        timeofday_to_label = json.load(f)

    # Create the backbone and detection model
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
        num_classes=len(category_to_label) + 1,  # background + classes
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    detection_model = detection_model.to(device)

    # Set attribute class counts from mapping files
    weather_classes = len(weather_to_label)
    scene_classes = len(scene_to_label)
    timeofday_classes = len(timeofday_to_label)
    print(f"Weather classes: {weather_classes}, Scene classes: {scene_classes}, Timeofday classes: {timeofday_classes}")

    # Instantiate the multi-task model and load saved weights
    multi_task_model = MultiTaskModel(detection_model, backbone,
                                      weather_classes=weather_classes,
                                      scene_classes=scene_classes,
                                      timeofday_classes=timeofday_classes).to(device)
    state_dict = torch.load("multi_task_model.pth", map_location=device)
    multi_task_model.load_state_dict(state_dict)
    multi_task_model.eval()
    print("Model loaded and ready for inference.")

    # Run inference on a sample image (update sample_image_path as needed)
    sample_image_path = "images/val/b1c9c847-3bda4659.jpg"
    output = infer_and_visualize(sample_image_path, multi_task_model, device, transform,
                                 category_to_label, weather_to_label, scene_to_label, timeofday_to_label)
    print("Inference output:")
    print(json.dumps(output, indent=2))
