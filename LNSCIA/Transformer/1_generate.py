import json
import random

# Dados Sintéticos mais Ricos
possible_labels = [
    "sinal de trânsito", "semáforo", "peão", "passadeira",
    "carro", "autocarro", "bicicleta", "ciclista", "estacionamento",
    "rua movimentada", "avenida", "estrada", "mota", "camião", "rotunda"
]

weather_conditions = ["ensolarado", "nublado", "chuvoso", "tempestuoso", "com nevoeiro", "vento forte"]
time_of_day = ["de manhã cedo", "no fim da manhã", "à tarde", "ao entardecer", "à noite", "de madrugada"]
locations = [
    "em uma rua movimentada", "no coração de uma grande avenida",
    "num bairro residencial tranquilo", "próximo a um parque verdejante",
    "perto de um shopping movimentado", "na periferia da cidade",
    "junto a uma escola", "perto de uma estação de metro"
]

traffic_density = ["trânsito leve", "trânsito moderado", "trânsito intenso", "engarrafamento"]
object_behaviors = [
    "em movimento rápido", "parados no semáforo", "aguardando para atravessar",
    "estacionados", "circulando lentamente", "cruzando a via"
]

templates = [
    "Num cenário {location}, percebe-se {objects}, todos {behavior}, sob condições de clima {weather} {time}, com {traffic}.",
    "Observa-se claramente {objects} {behavior}, situados {location}, num dia particularmente {weather} {time}, caracterizado por {traffic}.",
    "A cena mostra {objects}, atualmente {behavior}, em um ambiente {location}, sob um clima {weather} {time}, com {traffic} ao fundo.",
    "Durante um momento {weather} {time}, nota-se {objects} que estão {behavior} {location}, criando uma paisagem marcada por {traffic}.",
    "No contexto {location}, destacam-se {objects}, vistos {behavior}, numa altura do dia {weather} {time} e com {traffic}.",
    "Sob o céu {weather} {time}, {objects} podem ser observados {behavior} {location}, onde se percebe claramente {traffic}."
]

def generate_objects():
    num_objects = random.choice([2, 3, 4])
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
        traffic = random.choice(traffic_density)
        behavior = random.choice(object_behaviors)
        template = random.choice(templates)
        description = template.format(
            objects=objects,
            weather=weather,
            time=time,
            location=location,
            traffic=traffic,
            behavior=behavior
        )
        # Adiciona tokens especiais
        description = "startseq " + description + " endseq"
        labels = [obj.strip() for obj in objects.replace(",", " e ").split(" e ")]
        synthetic_example = {
            "labels": labels,
            "description": description,
            "origem": "sintético"
        }
        synthetic_data.append(synthetic_example)
    return synthetic_data

# Gerar e salvar os dados sintéticos
synthetic_data = generate_synthetic_data(1000)
with open('./data/synthetic_and_real_data.json', 'w', encoding='utf-8') as f:
    json.dump(synthetic_data, f, indent=4, ensure_ascii=False)

print("Dados sintéticos enriquecidos salvos com sucesso!")
