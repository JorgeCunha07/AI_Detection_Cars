import json
import random
import os

# Caminhos dos ficheiros
synthetic_file = './data/synthetic_data_realistic.json'
real_file = './data/real_data/real_data.json'
combined_file = './data/synthetic_and_real_data.json'

# Carregar dados sintéticos
synthetic_data = []
if os.path.exists(synthetic_file):
    with open(synthetic_file, 'r', encoding='utf-8') as f:
        synthetic_data = json.load(f)
else:
    print("Arquivo sintético não encontrado. Certifique-se de gerar os dados sintéticos primeiro.")
    exit(1)

# Carregar dados reais
real_data = []
if os.path.exists(real_file):
    with open(real_file, 'r', encoding='utf-8') as f:
        real_data = json.load(f)
else:
    print("Arquivo real não encontrado. Certifique-se de que o arquivo real_data.json existe.")
    exit(1)

# Combinar os dois conjuntos e embaralhar
combined_data = synthetic_data + real_data
random.shuffle(combined_data)

# Salvar o dataset combinado
with open(combined_file, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=4, ensure_ascii=False)

print("Conjunto combinado (sintético e real) salvo com sucesso em", combined_file)
