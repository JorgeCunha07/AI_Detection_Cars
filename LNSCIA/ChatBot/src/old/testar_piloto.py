import json
from transformers import pipeline

# Caminho para o modelo fine-tuned
model_path = "fine_tuning_model"

# Carrega o pipeline de Question Answering usando o modelo e tokenizer fine-tuned
qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)

# Carrega o contexto a partir do JSON convertido (ajuste o caminho se necessário)
with open("Codigo_Estrada_converted.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Concatena o campo "text" de todos os artigos para formar um grande contexto
context = ""
for article in data.get("articles", []):
    context += article.get("text", "") + "\n"

print("Context length:", len(context))

# Defina uma pergunta de exemplo (ajuste conforme necessário)
question = "Qual é a velocidade máxima em zonas urbanas?"

# Executa o pipeline para obter a resposta
result = qa_pipeline(question=question, context=context)

print("Question:", question)
print("Answer:", result["answer"])
