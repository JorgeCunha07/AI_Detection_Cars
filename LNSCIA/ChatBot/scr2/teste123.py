import json

def extrair_valores_unicos(caminho_arquivo):
    valores_weather = set()
    valores_scene = set()
    valores_timeofday = set()

    # Abre e carrega o arquivo JSON
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    # Itera sobre cada objeto na lista
    for item in dados:
        # Acessa o dicionário "attributes" se existir
        atributos = item.get("attributes", {})
        if "weather" in atributos:
            valores_weather.add(atributos["weather"])
        if "scene" in atributos:
            valores_scene.add(atributos["scene"])
        if "timeofday" in atributos:
            valores_timeofday.add(atributos["timeofday"])

    return valores_weather, valores_scene, valores_timeofday

if __name__ == '__main__':
    caminho = "bdd100k_labels_images_val.json"  # arquivo na mesma pasta do código
    weather, scene, timeofday = extrair_valores_unicos(caminho)
    
    print("Valores únicos para 'weather':", weather)
    print("Valores únicos para 'scene':", scene)
    print("Valores únicos para 'timeofday':", timeofday)
