import json
import random

# Possíveis objetos (rótulos) com nomes mais realistas
possible_labels = [
    "sinal de trânsito", "semáforo", "peão", "passadeira",
    "carro", "autocarro", "bicicleta", "ciclista", "estacionamento",
    "rua movimentada", "avenida", "estrada"
]

# Condições climáticas
# Unique values for weather: snowy, rainy, cloudy, undefined, foggy, partly cloudy, clear
# Valores únicos para o tempo: com neve, chuvoso, nublado, indefinido, nevoeiro, parcialmente nublado, limpo
weather_conditions = ["ensolarado", "nublado", "chuvoso", "tempestuoso", "com nevoeiro"]

# Horários do dia
# Unique values for timeofday: undefined, daytime, dawn/dusk, night
# Valores únicos para a hora do dia: indefinido, dia, amanhecer/entardecer, noite
time_of_day = ["de manhã", "à tarde", "à noite", "ao entardecer"]

# Locais
# Unique values for scene: motorway, car park, tunnel, undefined, service stations, residential, city street
# Valores únicos para o cenário: autoestrada, parque de estacionamento, túnel, indefinido, estações de serviço, residencial, rua da cidade
locations = [
    "em uma rua movimentada", "no coração de uma avenida",
    "num bairro residencial", "próximo a um parque",
    "perto de um shopping", "na periferia da cidade"
]

# Templates com linguagem mais fluida e variada
templates = [
    "Num cenário {location}, nota-se a presença de {objects} sob um clima {weather} {time}.",
    "Observa-se {objects} em um ambiente {location}, onde o dia se mostra {weather} {time}.",
    "A imagem capta {objects} situados {location}, com condições {weather} {time} ao fundo.",
    "Em meio a {location}, os elementos como {objects} se destacam num dia {weather} {time}.",
    "Durante um {weather} {time}, {objects} aparecem {location}, compondo uma cena singular."
]

def generate_objects():
    # Seleciona aleatoriamente 2 ou 3 objetos e os formata de maneira natural.
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
        
        # Monta a descrição utilizando os valores escolhidos de forma mais fluida
        description = template.format(
            objects=objects,
            weather=weather,
            time=time,
            location=location
        )
        
        # Insere os tokens especiais de início e fim de sequência
        description = "startseq " + description + " endseq"
        
        # Para os rótulos, extraímos os objetos utilizados na descrição
        labels = [obj.strip() for obj in objects.replace(",", " e ").split(" e ")]
        synthetic_example = {
            "labels": labels,
            "description": description
        }
        
        synthetic_data.append(synthetic_example)
    return synthetic_data

# Gerar e salvar os dados sintéticos com uma linguagem mais natural
synthetic_data = generate_synthetic_data(1000)

with open('./data/synthetic_data_realistic.json', 'w', encoding='utf-8') as file:
    json.dump(synthetic_data, file, indent=4, ensure_ascii=False)

print("Dados sintéticos realistas gerados com sucesso.")
