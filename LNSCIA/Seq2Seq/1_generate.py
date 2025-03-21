import json
import random

# Categorias organizadas claramente
categories = {
    "veiculos": ["carro", "autocarro", "camião"],
    "pessoas": ["peão", "ciclista"],
    "infraestrutura": ["passadeira", "semáforo", "passagem de nível"],
    "sinais_transito": ["limite de velocidade", "sinal de stop", "sinais de obrigação", "sinal de proibido", "obras na via"]
}

weather_conditions = ["céu limpo", "nublado", "chuvoso", "tempestuoso", "com nevoeiro", "vento forte"]
time_of_day = ["de manhã cedo", "a meio da manhã", "à tarde", "ao entardecer", "à noite", "de madrugada"]
traffic_density = ["trânsito leve", "trânsito moderado", "trânsito intenso", "engarrafamento"]

locations = [
    "numa rua movimentada", "no centro de uma grande avenida",
    "num bairro residencial tranquilo", "próximo de um parque",
    "perto de um centro comercial movimentado", "nos arredores da cidade",
    "junto a uma escola", "próximo de uma estação de metro"
]

object_behaviors = [
    "a circular rapidamente", "parados no semáforo", "a aguardar para atravessar",
    "estacionados", "a circular lentamente", "a atravessar a via", "em obras"
]

# Templates mais variados e em português de Portugal
templates = [
    "{time}, num dia {weather}, observam-se {veiculos} {behavior}, enquanto {pessoas} estão {pessoas_behavior}, {location}, com {traffic}.",
    "Durante a {time}, sob um clima {weather}, podem ver-se {pessoas} {pessoas_behavior}, bem como {infraestrutura} e {sinais_transito} visíveis {location}, além de {veiculos} {behavior}, com {traffic}.",
    "Com o tempo {weather}, {location}, é possível notar {veiculos} {behavior}, enquanto {pessoas} encontram-se {pessoas_behavior}. Também se destacam {infraestrutura} e {sinais_transito}, numa situação de {traffic}.",
    "{location}, {time}, sob um céu {weather}, destacam-se {veiculos} {behavior}, {pessoas} que estão {pessoas_behavior}, além de {infraestrutura} e {sinais_transito}, com {traffic}.",
    "Num ambiente {location}, os {veiculos} encontram-se {behavior}, enquanto os {pessoas} estão {pessoas_behavior}, num dia {weather} {time}, com {infraestrutura}, {sinais_transito} e {traffic}.",
    "A cena {weather}, situada {location} {time}, inclui {veiculos} {behavior}, {pessoas} {pessoas_behavior}, e infraestruturas como {infraestrutura} e {sinais_transito}, sob condições de {traffic}.",
    "É possível ver claramente {pessoas} {pessoas_behavior} e {veiculos} {behavior} {location}, com a presença de {infraestrutura} e {sinais_transito}, num dia {weather}, enfrentando {traffic}.",
    "Ao longo de {location}, a {time}, sob um céu {weather}, {pessoas} podem ser vistas {pessoas_behavior}, enquanto {veiculos} passam {behavior}, com {infraestrutura} e {sinais_transito} em destaque, numa situação de {traffic}.",
    "Uma imagem típica {location} mostra {veiculos} {behavior}, com {pessoas} {pessoas_behavior}, num dia {weather} {time}, acompanhado por {infraestrutura}, {sinais_transito} e {traffic}."
]

# Função que escolhe aleatoriamente elementos das categorias
def choose_from_category(category_name, min_items=1, max_items=2):
    items = categories[category_name]
    chosen_items = random.sample(items, k=random.randint(min_items, min(max_items, len(items))))
    return chosen_items

# Geração de dados sintéticos mais organizada e realista
def generate_synthetic_data(num_samples=1000):
    synthetic_data = []
    for _ in range(num_samples):
        veiculos = choose_from_category("veiculos")
        pessoas = choose_from_category("pessoas")
        infraestrutura = choose_from_category("infraestrutura")
        sinais_transito = choose_from_category("sinais_transito")

        weather = random.choice(weather_conditions)
        time = random.choice(time_of_day)
        location = random.choice(locations)
        traffic = random.choice(traffic_density)
        behavior = random.choice(object_behaviors)
        pessoas_behavior = random.choice(["a aguardar para atravessar", "a caminhar pela via", "à espera junto ao semáforo"])

        template = random.choice(templates)
        description = template.format(
            veiculos=", ".join(veiculos),
            pessoas=", ".join(pessoas),
            infraestrutura=", ".join(infraestrutura),
            sinais_transito=", ".join(sinais_transito),
            weather=weather,
            time=time,
            location=location,
            traffic=traffic,
            behavior=behavior,
            pessoas_behavior=pessoas_behavior
        )

        description = f"startseq {description} endseq"

        labels = veiculos + pessoas + infraestrutura + sinais_transito

        synthetic_example = {
            "labels": labels,
            "description": description,
            "origem": "sintético"
        }

        synthetic_data.append(synthetic_example)
    return synthetic_data

# Gerar dados sintéticos organizados
synthetic_data = generate_synthetic_data(100000)

# Salvar dados sintéticos
with open('./data/synthetic_and_real_data.json', 'w', encoding='utf-8') as f:
    json.dump(synthetic_data, f, indent=4, ensure_ascii=False)

print("Dados sintéticos organizados e realistas salvos com sucesso!")