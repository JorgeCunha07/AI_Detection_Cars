import json

# Caminho para o ficheiro JSON original
ficheiro_json = "data/frases_rodoviarias.json"

# Caminho para o novo ficheiro com descrições únicas
ficheiro_saida = "data/frases_rodoviarias.json"

# Carregar os dados
with open(ficheiro_json, "r", encoding="utf-8") as f:
    dados = json.load(f)

# Usar um conjunto para rastrear descrições únicas
descricoes_vistas = set()
dados_unicos = []

for item in dados:
    descricao = item.get("description")
    if descricao not in descricoes_vistas:
        descricoes_vistas.add(descricao)
        dados_unicos.append(item)

print(f"Total original: {len(dados)}")
print(f"Total após remoção de duplicados: {len(dados_unicos)}")

# Guardar os dados únicos num novo ficheiro
with open(ficheiro_saida, "w", encoding="utf-8") as f:
    json.dump(dados_unicos, f, ensure_ascii=False, indent=2)