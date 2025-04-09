import os
import json

def build_category_mapping(labels_dir):
    categories = set()
    for file_name in os.listdir(labels_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(labels_dir, file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Supondo que usamos o primeiro frame para extrair as categorias
                for obj in data["frames"][0]["objects"]:
                    categories.add(obj["category"])
    # Ordena as categorias para garantir consistência e mapeia (background = 0)
    categories = sorted(list(categories))
    category_to_label = {cat: i+1 for i, cat in enumerate(categories)}
    return category_to_label

# Exemplo de uso:
labels_directory = "../labels/train"  # caminho para os ficheiros JSON
detected_categories = build_category_mapping(labels_directory)
print("Categorias detectadas:", detected_categories)