import json
import random

# ---------------------------
# Parte 1: Geração de Dados Sintéticos
# ---------------------------
possible_labels = [
    "sinal de trânsito", "semáforo", "peão", "passadeira",
    "carro", "autocarro", "bicicleta", "ciclista", "estacionamento",
    "rua movimentada", "avenida", "estrada"
]

weather_conditions = ["ensolarado", "nublado", "chuvoso", "tempestuoso", "com nevoeiro"]
time_of_day = ["de manhã", "à tarde", "à noite", "ao entardecer"]
locations = [
    "em uma rua movimentada", "no coração de uma avenida",
    "num bairro residencial", "próximo a um parque",
    "perto de um shopping", "na periferia da cidade"
]

templates = [
    "Num cenário {location}, nota-se a presença de {objects} sob um clima {weather} {time}.",
    "Observa-se {objects} em um ambiente {location}, onde o dia se mostra {weather} {time}.",
    "A imagem capta {objects} situados {location}, com condições {weather} {time} ao fundo.",
    "Em meio a {location}, os elementos como {objects} se destacam num dia {weather} {time}.",
    "Durante um {weather} {time}, {objects} aparecem {location}, compondo uma cena singular."
]

def generate_objects():
    num_objects = random.choice([2, 3])
    selected = random.sample(possible_labels, k=num_objects)
    if num_objects == 2:
        return " e ".join(selected)
    else:
        return ", ".join(selected[:-1]) + " e " + selected[-1]

def generate_synthetic_data(num_samples=1000):
    synthetic_data = []
    for _ in range(num_samples):
        objects = generate_objects()
        weather = random.choice(weather_conditions)
        time = random.choice(time_of_day)
        location = random.choice(locations)
        template = random.choice(templates)
        description = template.format(
            objects=objects,
            weather=weather,
            time=time,
            location=location
        )
        description = "startseq " + description + " endseq"
        labels = [obj.strip() for obj in objects.replace(",", " e ").split(" e ")]
        synthetic_example = {
            "labels": labels,
            "description": description,
            "origem": "sintético"
        }
        synthetic_data.append(synthetic_example)
    return synthetic_data

# Gerar dados sintéticos
synthetic_data = generate_synthetic_data(1000)

# ---------------------------
# Parte 2: Carregar Dados Reais e Combinar
# ---------------------------
try:
    with open('./data/real_data/real_data.json', 'r', encoding='utf-8') as f:
        real_data = json.load(f)
except FileNotFoundError:
    print("Arquivo './data/real_data/real_data.json' não encontrado. Usando apenas dados sintéticos.")
    real_data = []

# Combinar os dados: os reais podem já estar no mesmo formato (com 'description') ou conter outros campos
combined_data = synthetic_data + real_data

# Salvar o conjunto combinado
with open('./data/synthetic_and_real_data.json', 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=4, ensure_ascii=False)

print("Conjunto combinado (dados sintéticos e reais) salvo com sucesso!")
