import os
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from pathlib import Path

print("🚀 A preparar o treino com ajustes de output e filtragem...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Dispositivo em uso: {device}")

chat_dir = Path(__file__).resolve().parent.parent / "chat"
modelo_output_dir = chat_dir / "gpt2-chat-finetuned"
dados_json_path = chat_dir / "dialogos.json"

# Carregar e filtrar dados
with open(dados_json_path, encoding="utf-8") as f:
    data = json.load(f)

seen = set()
filtrados = []
print(f"📚 Atntes da limpeza, total de exemplos: {len(data)}")
for d in data:
    k = (d.get("input", "").strip(), d.get("output", "").strip())
    if k not in seen and all(k):
        seen.add(k)
        filtrados.append({"input": k[0], "output": k[1]})

print(f"📚 Após limpeza, total de exemplos: {len(filtrados)}")

# Tokenizador e modelo
model_name = "pierreguillou/gpt2-small-portuguese"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Preparar textos
texts = [
    d["input"] + " " + tokenizer.eos_token + " " + d["output"]
    for d in filtrados
]
dataset = Dataset.from_dict({"text": texts})

# Tokenização

def tokenize(batch):
    result = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    result["labels"] = result["input_ids"].copy()
    return result

# Remover "text" para evitar erro de collation
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Argumentos
args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    report_to="none",
    remove_unused_columns=False
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(min(20, len(tokenized_dataset)))),
    data_collator=collator,
    tokenizer=tokenizer
)

print("🧠 A treinar com filtros e controle de output...")
trainer.train()

print("💾 A guardar modelo...")
modelo_output_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(modelo_output_dir)
tokenizer.save_pretrained(modelo_output_dir)
print(f"✅ Guardado em {modelo_output_dir}")
