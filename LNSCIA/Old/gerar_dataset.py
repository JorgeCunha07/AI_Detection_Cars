import csv
import random

# Listas de valores para cada categoria
tempos = ["manhã", "tarde", "noite", "dia"]
climas = ["ensolarado", "chuvoso", "nublado", "ventoso", "nevoeiro"]
veiculos = ["carros", "veículos"]
pedestres = ["peoes", "pedestres"]
passadeira = ["passadeira"]  # sempre a mesma
sinais = ["sinais de transito", "sinais"]

# Número de exemplos desejados
num_examples = 500

rows = []

for i in range(num_examples):
    # Seleciona valores aleatórios para tempo e clima
    tempo_val = random.choice(tempos)
    clima_val = random.choice(climas)
    
    # Cria uma lista de tags com base em probabilidades para cada categoria
    selected_tags = []
    if random.random() < 0.8:
        selected_tags.append(random.choice(veiculos))
    if random.random() < 0.7:
        selected_tags.append(random.choice(pedestres))
    if random.random() < 0.6:
        selected_tags.append(passadeira[0])
    if random.random() < 0.5:
        selected_tags.append(random.choice(sinais))
    
    # Sempre adiciona as informações de tempo e clima
    selected_tags.append("tempo " + tempo_val)
    selected_tags.append("clima " + clima_val)
    
    # Remove duplicatas e embaralha
    selected_tags = list(set(selected_tags))
    random.shuffle(selected_tags)
    vc_output = " ".join(selected_tags)
    
    # Cria a descrição com base nas tags selecionadas usando um template simples
    description_parts = []
    description_parts.append("Em")
    description_parts.append(tempo_val + ",")
    description_parts.append(clima_val + ",")
    
    if any(tag in vc_output for tag in veiculos):
        description_parts.append("os veículos trafegam")
    if any(tag in vc_output for tag in pedestres):
        description_parts.append("enquanto os pedestres")
        if "passadeira" in vc_output:
            description_parts.append("atravessam a passadeira")
        else:
            description_parts.append("caminham pela rua")
    else:
        if "passadeira" in vc_output:
            description_parts.append("a passadeira é utilizada")
    if any(tag in vc_output for tag in sinais):
        description_parts.append("sob a orientação dos sinais de trânsito")
    
    descricao = " ".join(description_parts) + "."
    
    rows.append((vc_output, descricao))

# Salva o dataset em um arquivo CSV
csv_filename = "dataset_exemplos_600.csv"
with open(csv_filename, mode="w", newline='', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["vc_output", "descricao"])
    writer.writerows(rows)

print(f"Dataset com {num_examples} exemplos salvo em '{csv_filename}'.")
