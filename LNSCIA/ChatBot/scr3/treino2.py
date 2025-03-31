import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import json

# Carrega o tokenizer e o modelo (a partir do diretório fine-tuned ou base)
model_dir = "./trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Forçar CPU
device = torch.device("cpu")
model.to(device)

# --- Preparação dos dados ---
# Supondo que tens um ficheiro JSON com exemplos para treino.
# Cada exemplo tem o formato: {"pergunta": "...", ... }
# Vamos construir os prompts usando a pergunta.
def load_training_data(codigo_file, questions_file):
    # Carregar código da estrada
    with open(codigo_file, encoding="utf-8") as f:
        codigo_data = json.load(f)
    
    # Carregar perguntas
    with open(questions_file, encoding="utf-8") as f:
        questions_data = json.load(f)
    
    examples = []
    
    # Criar exemplos a partir do código
    for article in codigo_data.get("articles", []):
        prompt = f"Explique o seguinte artigo do Código da Estrada: {article['article']}\nResposta: {article['text']}"
        examples.append({"text": prompt})
    
    # Criar exemplos a partir de perguntas e respostas
    for item in questions_data:
        for resposta in item.get("respostas_corretas", []):
            prompt = f"Pergunta: {item['pergunta']}\nResposta: {resposta}"
            examples.append({"text": prompt})
    
    return examples

# Carrega os dados do ficheiro de questões
train_examples = load_training_data("Codigo_Estrada_converted.json", "questions_dataset_enhanced.json")
dataset = Dataset.from_list(train_examples)

# Divide o dataset em treino (90%) e avaliação (10%)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Tokeniza os dados
def tokenize_function(example):
    # Define explicitamente um valor máximo (1024 é comum para GPT-2)
    return tokenizer(example["text"], truncation=True, max_length=1024)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator para LM (modelo causal)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- Configuração do treino com Trainer API ---
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,                  # Ajusta conforme necessário
    per_device_train_batch_size=2,       # Ajusta com base na memória disponível
    evaluation_strategy="steps",         # Avalia a cada eval_steps
    eval_steps=100,                      # Intervalo de avaliação (ajusta conforme o tamanho do dataset)
    logging_steps=50,
    save_steps=100,
    save_total_limit=2,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# Se pretender calcular alguma métrica adicional, pode definir a função compute_metrics
def compute_metrics(eval_pred):
    # Para modelos de geração, normalmente a loss já é a métrica principal.
    # Aqui, apenas devolvemos a loss.
    logits, labels = eval_pred
    return {"loss": float(torch.tensor(logits).mean().item())}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,       # Adiciona o conjunto de avaliação
    data_collator=data_collator,
    # compute_metrics=compute_metrics,  # Opcional: descomente se definir compute_metrics
)

# Inicia o treino
trainer.train()

# Salva o modelo final
trainer.save_model(model_dir)

def similar_improved(self, texto1, texto2, threshold=0.6):
    """Versão melhorada para comparar respostas"""
    # Pré-processamento: normalização
    texto1 = texto1.lower().strip()
    texto2 = texto2.lower().strip()
    
    # Remove pontuação e caracteres especiais
    import re
    texto1 = re.sub(r'[^\w\s]', '', texto1)
    texto2 = re.sub(r'[^\w\s]', '', texto2)
    
    # Lista de palavras comuns a ignorar
    stop_words = {'o', 'a', 'os', 'as', 'um', 'uma', 'de', 'da', 'do', 'das', 'dos', 'em', 'na', 'no', 'à', 'ao', 'e', 'é'}
    
    # Divide em palavras e remove stop words
    palavras1 = [p for p in texto1.split() if p not in stop_words]
    palavras2 = [p for p in texto2.split() if p not in stop_words]
    
    if not palavras1 or not palavras2:
        return False
    
    # Usando multisets (Counter) para considerar frequência de palavras
    from collections import Counter
    contador1 = Counter(palavras1)
    contador2 = Counter(palavras2)
    
    # Calcula a interseção considerando a frequência
    intersection = sum((contador1 & contador2).values())
    
    # Calcula a similaridade de Jaccard ponderada
    return intersection / (sum(contador1.values()) + sum(contador2.values()) - intersection) >= threshold
