import os
import sys
import json
import torch
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import Dataset
from pathlib import Path
import math
import torch
from torch.nn import CrossEntropyLoss

print("🚀 Preparando o treino com melhorias e Early Stopping...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Dispositivo em uso: {device}")

# Diretórios e caminhos
chat_dir = Path(__file__).resolve().parent.parent / "chat"
modelo_output_dir = chat_dir / "gpt2-chat-finetuned"
dados_json_path = chat_dir / "dialogos_validos3.json"

# Carregar e filtrar dados
with open(dados_json_path, encoding="utf-8") as f:
    data = json.load(f)

seen = set()
filtrados = []
print(f"🔍 Filtrando dados do ficheiro", dados_json_path)
print(f"📚 Antes da limpeza, total de exemplos: {len(data)}")
for d in data:
    inp = d.get("input", "").strip()
    out = d.get("output", "").strip()
    k = (inp, out)
    if k not in seen and inp and out:
        seen.add(k)
        filtrados.append({"input": inp, "output": out})
    else:
        print(f"⚠️  Exemplo duplicado ou inválido encontrado: {k}", file=sys.stderr, flush=True)
print(f"📚 Após limpeza, total de exemplos: {len(filtrados)}")

# Preparar os textos com marcadores explícitos
texts = [
    f"Usuário: {d['input']}\nAssistente: {d['output']}"
    for d in filtrados
]

# Criar dataset a partir dos textos
dataset = Dataset.from_dict({"text": texts})

# Inicializar tokenizador e modelo base
model_name = "pierreguillou/gpt2-small-portuguese"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Função de tokenização com loss masking para os tokens do prompt
def tokenize(batch):
    outputs = tokenizer(batch["text"], truncation=True, padding="longest", max_length=256)
    input_ids = outputs["input_ids"]
    labels = []
    assistant_marker = "Assistente: "
    for i, text in enumerate(batch["text"]):
        decoded = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        pos = decoded.find(assistant_marker)
        if pos != -1:
            prompt_text = text.split("Assistente:")[0] + "Assistente: "
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_length = len(prompt_tokens)
        else:
            prompt_length = len(input_ids[i])
            print(f"⚠️  Aviso: marcador '{assistant_marker.strip()}' não encontrado no exemplo: {decoded[:50]}...")
        example_labels = input_ids[i].copy()
        for j in range(prompt_length):
            example_labels[j] = -100
        labels.append(example_labels)
    outputs["labels"] = labels
    return outputs

# Dividir o dataset em treino e validação (80/20) e aplicar tokenização
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"].map(tokenize, batched=True, remove_columns=["text"])
val_dataset = split_dataset["test"].map(tokenize, batched=True, remove_columns=["text"])

# Função para calcular métricas de avaliação (perplexidade)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Converter para tensores PyTorch
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    
    # Filtrar apenas posições com labels válidos (-100 são tokens mascarados)
    active_loss = labels != -100
    
    # Remodelar logits e filtrar apenas posições relevantes
    active_logits = logits.view(-1, logits.size(-1))[active_loss.view(-1)]
    active_labels = labels.view(-1)[active_loss.view(-1)]
    
    # Calcular a perda com CrossEntropyLoss
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(active_logits, active_labels)
    
    # Limitar a perda para evitar valores astronômicos de perplexidade
    capped_loss = min(loss.item(), 10)  
    perplexity = math.exp(capped_loss)
    
    return {"perplexity": perplexity}


# Configuração dos argumentos de treino com early stopping e avaliação com perplexidade
args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",             # Avaliação ao final de cada época
    per_device_train_batch_size=2,           # Tamanho do batch de treino
    per_device_eval_batch_size=4,            # Tamanho do batch de validação
    gradient_accumulation_steps=2,           # Acumulação de gradientes para simular batch maior
    num_train_epochs=10,                     # Número máximo de épocas (o treino pode parar antes)
    warmup_steps=200,                        # Passos de aquecimento para o scheduler 
    logging_steps=50,                        # Passos para logar informações
    save_strategy="epoch",
    learning_rate=3e-5,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none",
    remove_unused_columns=False,
    load_best_model_at_end=True,             # Carrega o melhor modelo ao final
    metric_for_best_model="perplexity",
    greater_is_better=False
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("🧠 Iniciando treino com Early Stopping...")
trainer.train()

print("💾 Salvando modelo...")
modelo_output_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(modelo_output_dir)
tokenizer.save_pretrained(modelo_output_dir)
print(f"✅ Modelo salvo em {modelo_output_dir}")
