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

# Carrega o dataset
with open('perguntas_codigo_conducao_1000_equilibradas.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define o modelo e o tokenizer para português
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.to(device)  # Move o modelo para o dispositivo (GPU ou CPU)
model.eval()      # Modo avaliação

def get_sentence_embeddings(texts):
    """
    Retorna os embeddings das sentenças utilizando o vetor do token [CLS] em batch.
    """
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
    # Move os inputs para o dispositivo
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_dim)
    return cls_embeddings.cpu().numpy()  # Retorna para CPU como numpy array

def compute_similarity(vec1, vec2):
    """
    Calcula a similaridade do cosseno entre dois vetores.
    """
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

def train(train_data, thresholds=np.arange(0.5, 1.0, 0.01)):
    """
    Para cada entrada do conjunto de treino:
      - Pré-calcula os embeddings das respostas em batch.
      - Calcula o embedding agregado (média) das respostas corretas.
      - Para cada resposta (correta e incorreta), calcula:
          * sim_avg: similaridade entre o embedding da resposta e o embedding agregado.
          * sim_max: similaridade máxima entre o embedding da resposta e os embeddings individuais das respostas corretas.
      - Realiza grid search sobre os limiares para encontrar aqueles que maximizam a acurácia.
    
    Imprime o progresso para cada iteração e retorna os melhores limiares e acurácias obtidas.
    """
    total_entries = len(train_data)
    print(f"Iniciando pré-cálculo dos embeddings para {total_entries} entradas...")
    for idx, entry in enumerate(train_data, start=1):
        responses = entry['respostas']
        texts = [resp.replace(" (correta)", "").strip() for resp in responses]
        # Calcula os embeddings em batch utilizando GPU se disponível
        embeddings = get_sentence_embeddings(texts)
        # Define os labels: 1 para resposta correta, 0 para incorreta
        labels = [1 if "(correta)" in resp else 0 for resp in responses]
        entry['embeddings'] = embeddings
        entry['labels'] = labels

        # Exibe progresso a cada 10 entradas ou na última
        if idx % 10 == 0 or idx == total_entries:
            print(f"Pré-calculado embeddings para {idx}/{total_entries} entradas")

    best_threshold_avg = None
    best_acc_avg = -1
    best_threshold_max = None
    best_acc_max = -1

    total_thresholds = len(thresholds)
    print("Iniciando grid search pelos thresholds...")
    for i, thresh in enumerate(thresholds, 1):
        correct_avg = 0
        correct_max = 0
        total_examples = 0
        for entry in train_data:
            # Seleciona os embeddings das respostas corretas
            correct_embs = [emb for emb, label in zip(entry['embeddings'], entry['labels']) if label == 1]
            if len(correct_embs) == 0:
                continue
            # Calcula o embedding agregado (média) das respostas corretas
            agg_emb = np.mean(correct_embs, axis=0)
            for resp_emb, label in zip(entry['embeddings'], entry['labels']):
                sim_avg = compute_similarity(resp_emb, agg_emb)
                sims = [compute_similarity(resp_emb, correct_emb) for correct_emb in correct_embs]
                sim_max = max(sims) if sims else 0

                pred_avg = 1 if sim_avg >= thresh else 0
                pred_max = 1 if sim_max >= thresh else 0

                if pred_avg == label:
                    correct_avg += 1
                if pred_max == label:
                    correct_max += 1
                total_examples += 1

        acc_avg = correct_avg / total_examples if total_examples > 0 else 0
        acc_max = correct_max / total_examples if total_examples > 0 else 0

        # Imprime o progresso atual do grid search
        print(f"Epoch {i}/{total_thresholds} - Threshold: {thresh:.2f} | acc_avg: {acc_avg:.2f} | acc_max: {acc_max:.2f}")

        if acc_avg > best_acc_avg:
            best_acc_avg = acc_avg
            best_threshold_avg = thresh
        if acc_max > best_acc_max:
            best_acc_max = acc_max
            best_threshold_max = thresh

    return best_threshold_avg, best_threshold_max, best_acc_avg, best_acc_max

# Separa os dados: as 5 primeiras perguntas para teste; o restante para treino.
test_set = data[:5]
train_set = data[5:]

if __name__ == '__main__':
    best_threshold_avg, best_threshold_max, best_acc_avg, best_acc_max = train(train_set)
    print("\nTreino concluído:")
    print(f"Melhor limiar (método avg): {best_threshold_avg:.2f} com acurácia: {best_acc_avg:.2f}")
    print(f"Melhor limiar (método max): {best_threshold_max:.2f} com acurácia: {best_acc_max:.2f}")
    # Salva os thresholds encontrados em um arquivo para uso no teste
    thresholds = {
        "threshold_avg": best_threshold_avg,
        "threshold_max": best_threshold_max
    }
    with open('thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=4)
