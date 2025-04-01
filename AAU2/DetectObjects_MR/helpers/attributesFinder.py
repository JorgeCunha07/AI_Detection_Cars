import os
import json

def build_attributes_mapping(labels_dir):
    weather_values = set()
    scene_values = set()
    timeofday_values = set()
    for file_name in os.listdir(labels_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(labels_dir, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                attrs = data.get("attributes", {})
                if "weather" in attrs:
                    weather_values.add(attrs["weather"])
                if "scene" in attrs:
                    scene_values.add(attrs["scene"])
                if "timeofday" in attrs:
                    timeofday_values.add(attrs["timeofday"])
    weather_to_label = {val: i for i, val in enumerate(sorted(list(weather_values)))}
    scene_to_label = {val: i for i, val in enumerate(sorted(list(scene_values)))}
    timeofday_to_label = {val: i for i, val in enumerate(sorted(list(timeofday_values)))}
    return {
         "weather": weather_to_label,
         "scene": scene_to_label,
         "timeofday": timeofday_to_label
    }

# Exemplo de uso:
labels_directory = "../labels/train"  # caminho para os ficheiros JSON
detected_categories = build_attributes_mapping(labels_directory)
print("Categorias detectadas:", detected_categories)