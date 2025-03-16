import csv
import os

# Lista de exemplos: (palavras-chave, descrição)
dataset = [
    ("peão passadeira semáforo dia chuva", "Em um dia chuvoso, o peão atravessa a passadeira sob o semáforo."),
    ("peão passadeira semáforo dia sol", "Num dia ensolarado, o peão utiliza a passadeira com atenção ao semáforo."),
    ("passadeira peão semáforo noite chuva", "Durante a chuva à noite, o peão cruza a passadeira observando o semáforo."),
    ("peão semáforo passadeira dia nublado", "Em um dia nublado, o peão atravessa a passadeira enquanto o semáforo indica cautela."),
    ("passadeira semáforo peão dia chuva", "Mesmo em um dia de chuva, o peão segue pela passadeira enquanto o semáforo permanece vermelho."),
    ("peão passadeira semáforo dia tempestade", "Durante uma tempestade, o peão atravessa a passadeira, atento ao semáforo."),
    ("passadeira semáforo peão dia ensolarado", "Em um dia ensolarado, o peão atravessa a passadeira quando o semáforo está verde."),
    ("peão semáforo passadeira tarde chuva", "Na tarde chuvosa, o peão cruza a passadeira seguindo as orientações do semáforo."),
    ("passadeira peão semáforo dia chuva", "Num dia chuvoso, o peão aguarda o semáforo e atravessa a passadeira com segurança."),
    ("peão semáforo passadeira dia sol", "Em um dia de sol, o peão atravessa a passadeira sob a luz do semáforo verde.")
]

# Cria a pasta de destino, se não existir
output_dir = os.path.join("data", "processed")
os.makedirs(output_dir, exist_ok=True)

# Define o caminho do arquivo
output_file = os.path.join(output_dir, "synthetic_dataset.tsv")

# Escreve os dados no arquivo TSV
with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    # Cabeçalho
    writer.writerow(["input_keywords", "output_description"])
    # Dados
    for inp, out in dataset:
        writer.writerow([inp, out])

print("Dataset sintético salvo em", output_file)