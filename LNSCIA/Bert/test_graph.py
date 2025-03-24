import torch
from transformers import BertTokenizer, BertModel

# Define o dispositivo: GPU se disponível, senão CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device)

# Nome do modelo e carregamento do tokenizer e do modelo
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.to(device)
model.eval()

# Cria uma entrada dummy para traçar o modelo
dummy_text = "Exemplo de texto para traçar o modelo."
inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
# Move os inputs para o dispositivo (GPU ou CPU)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Converte o modelo para TorchScript utilizando torch.jit.trace
# O modelo recebe 'input_ids' e 'attention_mask' como argumentos posicionais
traced_model = torch.jit.trace(model, (inputs["input_ids"], inputs["attention_mask"]))
# Salva o modelo traçado
traced_model.save("bert_traced.pt")
print("Modelo TorchScript salvo como bert_traced.pt")

# --- Exemplo de carregamento e inferência com o modelo traçado ---

# Carrega o modelo traçado
loaded_model = torch.jit.load("bert_traced.pt")
loaded_model.to(device)
loaded_model.eval()

# Teste de inferência: extrai o embedding do token [CLS] para uma sentença de exemplo
test_text = "Este é um exemplo de teste para o modelo BERT."
test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
test_inputs = {k: v.to(device) for k, v in test_inputs.items()}

with torch.no_grad():
    # Executa a inferência utilizando o modelo traçado
    outputs = loaded_model(test_inputs["input_ids"], test_inputs["attention_mask"])

# O embedding do token [CLS] está na posição 0 da sequência
cls_embedding = outputs.last_hidden_state[:, 0, :]
print("Embedding do token [CLS]:", cls_embedding)
