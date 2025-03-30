#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script completo para treinar o BERTimbau em uma tarefa de Question Answering,
utilizando os dados contidos em arquivos JSON nas novas pastas:
  data/Diario_da_Republica/  -> Codigo_Estrada.json
  data/Questoes/             -> perguntas_codigo_conducao_1000_equilibradas.json

Estrutura de pastas esperada:
data/
  Bom_Condutor/
  Diario_da_Republica/
    Codigo_Estrada.json
  Questoes/
    perguntas_codigo_conducao_1000_equilibradas.json

notebooks/

src/
  __init__.py
  model.py
  test_gpu.py
  train.py

Para executar:
  python src/train.py
"""

import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# -------------------------------------------------------------------
# 1. Carregando o contexto do Código da Estrada
# -------------------------------------------------------------------
def load_code_context(json_file):
    """
    Lê o arquivo JSON do Código da Estrada e concatena todo o conteúdo
    em uma única string para servir como 'contexto' das perguntas.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    context_text = ""
    # Assume que data["conteudo"] é uma lista onde cada item possui:
    #   "titulo" e uma lista de "capitulos"
    for item in data.get("conteudo", []):
        context_text += item.get("titulo", "") + "\n"
        for cap in item.get("capitulos", []):
            context_text += cap.get("capitulo", "") + "\n"
            for art in cap.get("artigos", []):
                context_text += art.get("artigo", "") + "\n" + art.get("conteudo", "") + "\n"
    return context_text


# -------------------------------------------------------------------
# 2. Carregando as questões
# -------------------------------------------------------------------
def load_questions(json_file):
    """
    Lê o arquivo JSON que contém perguntas e respostas. 
    Seleciona apenas a primeira resposta que estiver marcada como '(correta)'.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = []
    # Assume que as questões estão em data["conteudo"]["questoes"]
    for q in data.get("conteudo", {}).get("questoes", []):
        question_text = q.get("pergunta", "").strip()

        # Seleciona a primeira resposta marcada como '(correta)'
        correct_answer = None
        for resp in q.get("respostas", []):
            if "(correta)" in resp:
                correct_answer = resp.replace("(correta)", "").strip()
                break

        # Se não encontrar nenhuma resposta marcada, pula a questão
        if not correct_answer:
            continue

        questions.append({
            "id": q.get("id"),
            "question": question_text,
            "answer_text": correct_answer
        })

    return questions


# -------------------------------------------------------------------
# 3. Definição dos caminhos de acordo com a nova estrutura
# -------------------------------------------------------------------
# Ajuste se necessário, dependendo de onde você executa o script
DATA_DIR = os.path.join("data")
CODIGO_FILE = os.path.join(DATA_DIR, "Bom_Condutor", "Codigo_Estrada.json")
QUESTIONS_FILE = os.path.join(DATA_DIR, "Questoes", "perguntas_codigo_conducao_1000_equilibradas.json")


# -------------------------------------------------------------------
# 4. Carrega o contexto (Código da Estrada) e as perguntas
# -------------------------------------------------------------------
print("Carregando contexto do Código da Estrada...")
context = load_code_context(CODIGO_FILE)

print("Carregando questões...")
questions = load_questions(QUESTIONS_FILE)

# Cria um dataset do Hugging Face – cada pergunta recebe o mesmo contexto
dataset_dict = {
    "id": [q["id"] for q in questions],
    "question": [q["question"] for q in questions],
    "context": [context for _ in questions],
    "answer_text": [q["answer_text"] for q in questions]
}
dataset = Dataset.from_dict(dataset_dict)


# -------------------------------------------------------------------
# 5. Função para calcular o índice inicial da resposta dentro do contexto
# -------------------------------------------------------------------
def add_answer_start(example):
    """
    Localiza a posição (índice inicial) da resposta no contexto.
    Caso não encontre, define como 0.
    """
    answer = example["answer_text"]
    start_idx = example["context"].find(answer)
    if start_idx == -1:
        start_idx = 0
    example["answer_start"] = start_idx
    return example

dataset = dataset.map(add_answer_start)


# -------------------------------------------------------------------
# 6. Tokenização e preparo dos spans
# -------------------------------------------------------------------
print("Carregando tokenizer do BERTimbau...")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

max_length = 384
doc_stride = 64


def preprocess_function(examples):
    """
    Tokeniza perguntas e contextos, e localiza os índices de início/fim
    dos spans de resposta no tokenizado.
    """
    questions_list = [q.strip() for q in examples["question"]]
    contexts_list = examples["context"]
    answers = [{"text": a, "answer_start": s} for a, s in zip(examples["answer_text"], examples["answer_start"])]

    # Tokenização
    tokenized_examples = tokenizer(
        questions_list,
        contexts_list,
        truncation="only_second",  # Trunca apenas o contexto
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Mapeamento para tratar splits de contexto (overflow)
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answer = answers[sample_index]
        answer_text = answer["text"]
        answer_start = answer["answer_start"]
        answer_end = answer_start + len(answer_text)

        sequence_ids = tokenized_examples.sequence_ids(i)
        # Encontra região do contexto (em BERT, geralmente é '1')
        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        # Se a resposta não está totalmente contida no trecho tokenizado
        if not (offsets[context_start][0] <= answer_start and offsets[context_end][1] >= answer_end):
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # Localiza o token de início
            token_start_index = context_start
            while token_start_index <= context_end and offsets[token_start_index][0] <= answer_start:
                token_start_index += 1
            token_start_index -= 1

            # Localiza o token de fim
            token_end_index = context_end
            while token_end_index >= context_start and offsets[token_end_index][1] >= answer_end:
                token_end_index -= 1
            token_end_index += 1

            tokenized_examples["start_positions"].append(token_start_index)
            tokenized_examples["end_positions"].append(token_end_index)

    return tokenized_examples

print("Tokenizando dataset...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=16,  # ou 8, ou até 4, dependendo da sua máquina
    remove_columns=["id", "question", "context", "answer_text", "answer_start"]
)

# Divide em treino e validação (exemplo: 90% / 10%)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)


# -------------------------------------------------------------------
# 7. Carrega o modelo e configura o Trainer
# -------------------------------------------------------------------
print("Carregando modelo BERTimbau para QA...")
model = AutoModelForQuestionAnswering.from_pretrained("neuralmind/bert-base-portuguese-cased")

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="outputs",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,       # aumenta o intervalo de log (opcional)
    save_steps=1000,        # salva com menos frequência
    save_total_limit=2,     # mantém apenas os 2 checkpoints mais recentes
    fp16=True,
    push_to_hub=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer
)


# -------------------------------------------------------------------
# 8. Inicia o treinamento e salva o modelo
# -------------------------------------------------------------------
print("Iniciando treinamento...")
trainer.train()

print("Salvando modelo fine-tuned...")
trainer.save_model("fine_tuning_model")

print("Treinamento concluído com sucesso!")

#{'eval_loss': 0.011700717732310295, 'eval_runtime': 141.7603, 'eval_samples_per_second': 110.68, 'eval_steps_per_second': 13.84, 'epoch': 3.0}                         
#{'train_runtime': 15904.53, 'train_samples_per_second': 26.634, 'train_steps_per_second': 3.329, 'train_loss': 0.017584750579170826, 'epoch': 3.0}                    
#100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52953/52953 [4:25:04<00:00,  3.33it/s] 
#Salvando modelo fine-tuned...
#Treinamento concluído com sucesso!