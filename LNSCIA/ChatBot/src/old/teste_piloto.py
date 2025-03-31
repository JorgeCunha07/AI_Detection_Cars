#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para teste piloto do treinamento (fine-tuning) do modelo BERTimbau para Question Answering.

Este script:
  1. Carrega um dataset piloto a partir de um arquivo JSON.
  2. Tokeniza o dataset usando o tokenizer do modelo "neuralmind/bert-base-portuguese-cased".
  3. Se houver mais de 1 exemplo, divide o dataset em treino e validação; caso contrário, usa o mesmo exemplo para ambos.
  4. Executa um treinamento piloto (1 época) para validar o pipeline.
  
O script só prossegue com o treinamento se detectar (por exemplo) a presença de GPU (neste exemplo, assume-se que a mensagem de GPU na saída indica que a GPU está ativa).

Para executar:
  python teste_piloto.py
"""

import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# ============================================================================
# 1. Função para carregar o dataset piloto
# ============================================================================
def load_pilot_data(json_file):
    """
    Lê o arquivo JSON que contém um objeto com chave "questions" (lista de exemplos)
    Cada exemplo deve conter:
      - "question": texto da pergunta
      - "context": contexto a ser utilizado
      - "answer_text": a resposta correta (texto)
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Se o arquivo tiver a chave "questions", usa-a; caso contrário, assume que data é a lista
    questions = data.get("questions", data)
    dataset_dict = {
        "question": [ex.get("question", "").strip() for ex in questions],
        "context": [ex.get("context", "").strip() for ex in questions],
        "answer_text": [ex.get("answer_text", "").strip() for ex in questions]
    }
    return Dataset.from_dict(dataset_dict)

# ============================================================================
# 2. Função de pré-processamento (tokenização) para o dataset piloto
# ============================================================================
def preprocess_pilot(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",  # Trunca apenas o contexto se necessário
        max_length=384,
        stride=64,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    # Recupera mapeamento para tratar splits de contexto (overflow)
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    # Para cada tokenizado, calcula a posição de início/fim da resposta
    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answer_text = examples["answer_text"][sample_index]
        context_text = examples["context"][sample_index]
        answer_start = context_text.find(answer_text)
        answer_end = answer_start + len(answer_text)
        # Obter os IDs de sequência para identificar a parte do contexto (em geral 1)
        sequence_ids = tokenized.sequence_ids(i)
        # Localiza a posição do início e fim do contexto no tokenizado
        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        # Se a resposta não está totalmente contida no trecho tokenizado, define posições como 0
        if not (offsets[context_start][0] <= answer_start and offsets[context_end][1] >= answer_end):
            tokenized["start_positions"].append(0)
            tokenized["end_positions"].append(0)
        else:
            # Localiza o token que marca o início da resposta
            token_start_index = context_start
            while token_start_index <= context_end and offsets[token_start_index][0] <= answer_start:
                token_start_index += 1
            token_start_index -= 1

            # Localiza o token que marca o fim da resposta
            token_end_index = context_end
            while token_end_index >= context_start and offsets[token_end_index][1] >= answer_end:
                token_end_index -= 1
            token_end_index += 1

            tokenized["start_positions"].append(token_start_index)
            tokenized["end_positions"].append(token_end_index)

    return tokenized

# ============================================================================
# 3. Carregando modelo e tokenizer
# ============================================================================
print("Carregando tokenizer e modelo: neuralmind/bert-base-portuguese-cased")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModelForQuestionAnswering.from_pretrained("neuralmind/bert-base-portuguese-cased")

# ============================================================================
# 4. Carregando o dataset piloto
# ============================================================================
pilot_file = "pilot_dataset.json"  # ajuste o caminho se necessário
if not os.path.exists(pilot_file):
    print(f"Arquivo {pilot_file} não encontrado. Criando exemplo de teste piloto.")
    pilot_data = {
        "questions": [
            {
                "question": "Qual é a velocidade máxima em zonas urbanas?",
                "context": "De acordo com o Código da Estrada, a velocidade máxima em zonas urbanas é de 50 km/h.",
                "answer_text": "50 km/h"
            }
        ]
    }
    with open(pilot_file, "w", encoding="utf-8") as f:
        json.dump(pilot_data, f, ensure_ascii=False, indent=2)

print("Carregando dataset piloto...")
pilot_dataset = load_pilot_data(pilot_file)
print("Número de exemplos no dataset piloto:", len(pilot_dataset))

# ============================================================================
# 5. Tokenização do dataset piloto
# ============================================================================
print("Tokenizando dataset piloto...")
pilot_tokenized = pilot_dataset.map(
    preprocess_pilot,
    batched=True,
    batch_size=2,
    remove_columns=pilot_dataset.column_names
)
print("Número de exemplos após tokenização:", len(pilot_tokenized))

# Se houver mais de 1 exemplo, divide em treino/validação; caso contrário, usa o mesmo para ambos.
if len(pilot_tokenized) > 1:
    split_dataset = pilot_tokenized.train_test_split(test_size=0.5)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print("Exemplos no treino:", len(train_dataset))
    print("Exemplos na avaliação:", len(eval_dataset))
else:
    train_dataset = pilot_tokenized
    eval_dataset = pilot_tokenized
    print("Apenas 1 exemplo disponível; usando-o para treino e avaliação.")

# ============================================================================
# 6. Configuração dos TrainingArguments e criação do Trainer
# ============================================================================
training_args = TrainingArguments(
    output_dir="outputs_piloto",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# ============================================================================
# 7. Inicia o treinamento piloto
# ============================================================================
print("Iniciando treinamento piloto...")
trainer.train()
print("Treinamento piloto concluído com sucesso!")
