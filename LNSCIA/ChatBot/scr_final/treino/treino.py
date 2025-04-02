import os
import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from pathlib import Path

print("🚀 A preparar o treino do modelo GPT2 com os teus diálogos absurdamente legais...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Dispositivo em uso: {device}")

# Caminhos
chat_dir = Path(__file__).resolve().parent.parent / "chat"
modelo_output_dir = chat_dir / "gpt2-chat-finetuned"
dados_json_path = chat_dir / "dialogos.json"

# Carregamento dos dados
with open(dados_json_path, encoding="utf-8") as f:
    data = json.load(f)
print(f"📚 Exemplos carregados: {len(data)}")

# Debug das chaves
print("🔎 Exibindo as chaves dos 5 primeiros exemplos:")
for i in range(min(5, len(data))):
    print(f"Exemplo {i}: {list(data[i].keys())}")

# Tokenizador e modelo
model_name = "pierreguillou/gpt2-small-portuguese"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Preparar textos
texts = [
    d["input"] + " " + tokenizer.eos_token + " " + d["output"]
    for d in data if "input" in d and "output" in d
]
print(f"📝 Nº de textos após filtragem: {len(texts)}")
dataset = Dataset.from_dict({"text": texts})
print("✅ Dataset construído com sucesso.")

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

# Removemos a coluna "text" para evitar que o DataCollator tente processá-la
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
print("🔍 Dataset tokenizado. Nº de amostras:", len(tokenized_dataset))

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

# Collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(min(10, len(tokenized_dataset)))),
    data_collator=collator,
    tokenizer=tokenizer
)

# Treinar
print("🧠 A treinar o modelo...")
trainer.train()

# Guardar modelo
print("💾 A guardar o modelo treinado...")
modelo_output_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(modelo_output_dir)
tokenizer.save_pretrained(modelo_output_dir)
print(f"✅ Modelo guardado em {modelo_output_dir}")
