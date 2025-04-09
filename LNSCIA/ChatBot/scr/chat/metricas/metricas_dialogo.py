import json
from transformers import GPT2Tokenizer
from pathlib import Path
import statistics

# Caminho para o teu ficheiro
caminho_ficheiro = Path(__file__).resolve().parent / "dialogos_validos4.json"

# Carregar dados
with open(caminho_ficheiro, "r", encoding="utf-8") as f:
    dados = json.load(f)

# Carregar tokenizer do modelo fine-tuned (ou genérico GPT-2 se quiseres testar)
tokenizer = GPT2Tokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese")

# Listas para armazenar os valores
num_palavras_input = []
num_palavras_output = []
num_tokens_input = []
num_tokens_output = []

for item in dados:
    pergunta = item["input"]
    resposta = item["output"]

    # Contar palavras
    num_palavras_input.append(len(pergunta.split()))
    num_palavras_output.append(len(resposta.split()))

    # Contar tokens
    num_tokens_input.append(len(tokenizer.encode(pergunta)))
    num_tokens_output.append(len(tokenizer.encode(resposta)))

# Calcular médias
def media(lista):
    return round(statistics.mean(lista), 2)

print("📊 Estatísticas do dataset:")
print(f"🔹 Média de palavras nas perguntas: {media(num_palavras_input)}")
print(f"🔹 Média de palavras nas respostas: {media(num_palavras_output)}")
print(f"🔹 Média de tokens nas perguntas: {media(num_tokens_input)}")
print(f"🔹 Média de tokens nas respostas: {media(num_tokens_output)}")
print(f"🔹 Total de pares analisados: {len(dados)}")
