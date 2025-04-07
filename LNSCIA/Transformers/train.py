from transformers import MT5ForConditionalGeneration, TrainingArguments, Trainer
from pre_processing import load_and_preprocess, tokenizer
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
import torch

print("🖥️  Dispositivo ativo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Carregar dataset tokenizado
tokenized_dataset = load_and_preprocess()
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Carregar modelo base
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

# Data collator que trata do padding dos labels
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

# Argumentos de treino
training_args = TrainingArguments(
    output_dir="./mt5-descriptions",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=50,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Criar trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

# Guardar modelo final
model.save_pretrained("./modelo-final")
tokenizer.save_pretrained("./modelo-final")
