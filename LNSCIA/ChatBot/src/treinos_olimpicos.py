import os
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Verifica se GPU está disponível
if not torch.cuda.is_available():
    print("GPU não encontrada. Este script foi configurado para treinar apenas com GPU.")
    sys.exit(1)

# -----------------------------
# 1) Configurações iniciais
# -----------------------------
MODEL_NAME = "pierreguillou/gpt2-small-portuguese"  
PATH_QA = "qa_perguntas_respostas.json"  # Ficheiro com as Q&A
OUTPUT_DIR = "./meu_modelo_ajustado"

# -----------------------------
# 2) Função para carregar e preparar dados
# -----------------------------
def load_qa_data(path_qa):
    with open(path_qa, "r", encoding="utf-8") as f:
        data = json.load(f)
    qa_list = data["qa_examples"]
    input_texts = []
    target_texts = []
    for qa in qa_list:
        question = qa["question"]
        answer = qa["answer"]
        prompt = f"Pergunta: {question}\nResposta:"
        completion = f" {answer}"
        input_texts.append(prompt)
        target_texts.append(completion)
    return input_texts, target_texts

# -----------------------------
# 3) Criação do Dataset
# -----------------------------
def create_dataset(tokenizer, input_texts, target_texts, max_length=256):
    full_texts = [inp + tgt for inp, tgt in zip(input_texts, target_texts)]
    dataset_dict = {"text": full_texts}
    ds = Dataset.from_dict(dataset_dict)
    
    def tokenize(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    ds_tokenized = ds.map(tokenize, batched=True)
    return ds_tokenized

# -----------------------------
# 4) Script principal
# -----------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    input_texts, target_texts = load_qa_data(PATH_QA)
    ds_tokenized = create_dataset(tokenizer, input_texts, target_texts, max_length=256)
    
    ds_split = ds_tokenized.train_test_split(test_size=0.1, seed=42)
    ds_train = ds_split["train"]
    ds_val = ds_split["test"]
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=5e-5,
        fp16=True,  # Usará FP16, pois GPU está disponível
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Treino concluído! Modelo e tokenizer guardados em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
