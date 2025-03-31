from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Carrega o modelo fine-tuned e o tokenizer
model = AutoModelForQuestionAnswering.from_pretrained("fine_tuning_model")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Exemplo de contexto e pergunta
context = "Seu texto de contexto completo, por exemplo, o conteúdo do Código da Estrada."
question = "Qual é a velocidade máxima em zonas urbanas?"

# Tokeniza a entrada
inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)

# Obtém os índices de início e fim da resposta
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

# Converte os tokens em string
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
print("Resposta:", answer)
