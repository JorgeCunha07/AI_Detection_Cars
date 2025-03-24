import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define o dispositivo: GPU se disponível, senão CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado:", device)

# Carrega os thresholds salvos no treino
with open('thresholds.json', 'r', encoding='utf-8') as f:
    thresholds = json.load(f)
threshold_avg = thresholds["threshold_avg"]
threshold_max = thresholds["threshold_max"]
print(f"Threshold (avg): {threshold_avg:.2f}")
print(f"Threshold (max): {threshold_max:.2f}")

# Carrega o dataset e define o conjunto de teste (5 primeiras entradas)
with open('perguntas_codigo_conducao_1000_equilibradas.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
test_set = data[:5]

# Carrega o modelo e o tokenizer para português
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.to(device)
model.eval()

def get_sentence_embedding(text):
    """
    Retorna o embedding do token [CLS] para uma sentença.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Retorna o embedding do token [CLS]
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def compute_similarity(vec1, vec2):
    """
    Calcula a similaridade do cosseno entre dois vetores.
    """
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

# Inicializa contadores para avaliação
total_examples = 0
correct_avg_total = 0
correct_max_total = 0

print("\nIniciando avaliação no conjunto de teste...")
# Para cada entrada do conjunto de teste:
for entry in test_set:
    responses = entry["respostas"]
    # Extrai as respostas corretas e seus textos (removendo a marcação)
    correct_texts = [resp.replace(" (correta)", "").strip() for resp in responses if "(correta)" in resp]
    if len(correct_texts) == 0:
        continue
    # Calcula os embeddings para as respostas corretas e o embedding agregado (média)
    correct_embs = [get_sentence_embedding(text) for text in correct_texts]
    agg_emb = np.mean(correct_embs, axis=0)

    # Para cada resposta, calcula a similaridade e a previsão com os thresholds
    for resp in responses:
        label = 1 if "(correta)" in resp else 0
        text = resp.replace(" (correta)", "").strip()
        resp_emb = get_sentence_embedding(text)
        
        sim_avg = compute_similarity(resp_emb, agg_emb)
        sims = [compute_similarity(resp_emb, emb) for emb in correct_embs]
        sim_max = max(sims) if sims else 0

        pred_avg = 1 if sim_avg >= threshold_avg else 0
        pred_max = 1 if sim_max >= threshold_max else 0

        total_examples += 1
        if pred_avg == label:
            correct_avg_total += 1
        if pred_max == label:
            correct_max_total += 1

# Calcula e exibe as acurácias
accuracy_avg = correct_avg_total / total_examples if total_examples > 0 else 0
accuracy_max = correct_max_total / total_examples if total_examples > 0 else 0
print(f"\nAcurácia no conjunto de teste (método avg): {accuracy_avg:.2f}")
print(f"Acurácia no conjunto de teste (método max): {accuracy_max:.2f}")
