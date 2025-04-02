import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, default_data_collator
)
from datasets import Dataset
import json
import os
import shutil
import wandb
from colorama import Fore, Style, init

init(autoreset=True)

if not torch.cuda.is_available():
    print(f"{Fore.RED}ERRO: GPU não encontrada! Abortando.{Style.RESET_ALL}")
    exit(1)

device = torch.device("cuda")
print(f"{Fore.GREEN}Usando GPU: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")

# Modelo
model_name = "pierreguillou/gpt2-small-portuguese"
print(f"{Fore.CYAN}Carregando modelo base {model_name}...{Style.RESET_ALL}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # <- NÃO É MAIS FP16
    device_map="auto"
)

# Pad token
if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    print(f"{Fore.YELLOW}Adicionando pad token customizado...{Style.RESET_ALL}")
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
print(f"{Fore.YELLOW}Pad token ID: {tokenizer.pad_token_id}{Style.RESET_ALL}")

# Carregamento dos dados
try:
    with open("../../Documentacao/BomCondutor/Codigo_Estrada.json", encoding="utf-8") as f:
        codigo_data = json.load(f)
    print(f"{Fore.GREEN}Dados do Código da Estrada carregados com sucesso.{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Erro ao carregar Codigo_Estrada.json: {e}{Style.RESET_ALL}")
    codigo_data = {"articles": []}

# Criação dos exemplos
codigo_exemplos = []
for article in codigo_data.get("articles", []):
    prompt = f"<USER>: Explique o {article['article']} do Código da Estrada.\n<ASSISTANT>: {article['text']}"
    if 30 < len(prompt) < 1000:
        codigo_exemplos.append({"text": prompt})
print(f"{Fore.CYAN}Total de exemplos para treino: {len(codigo_exemplos)}{Style.RESET_ALL}")

dataset = Dataset.from_list(codigo_exemplos)

# Tokenização
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Adiciona labels = input_ids (necessário para treino CausalLM)
def add_labels(example):
    example["labels"] = example["input_ids"]
    return example

tokenized_dataset = tokenized_dataset.map(add_labels)

# Split
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"{Fore.CYAN}Exemplo de validação:{Style.RESET_ALL}")
print(eval_dataset[0])

# Configuração de treino
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    eval_strategy="steps",
    eval_steps=25,
    logging_steps=10,
    save_steps=25,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    fp16=False,  # <- AGORA DESLIGADO
    report_to="wandb"
)

# WandB
wandb.init(project="codigo-estrada-chatbot", reinit=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

# Treino
try:
    print(f"{Fore.GREEN}Iniciando treino...{Style.RESET_ALL}")
    trainer.train()
    print(f"{Fore.CYAN}Salvando modelo...{Style.RESET_ALL}")
    trainer.save_model("./trained_model")
    tokenizer.save_pretrained("./trained_model")
except Exception as e:
    print(f"{Fore.RED}Erro durante o treino: {e}{Style.RESET_ALL}")
    raise e
finally:
    torch.cuda.empty_cache()
    wandb.finish()

# Limpeza opcional
def limpar_ficheiros_temporarios():
    print(f"{Fore.CYAN}Limpando ficheiros temporários...{Style.RESET_ALL}")
    if os.path.exists("./trained_model"):
        for item in os.listdir("./trained_model"):
            path = os.path.join("./trained_model", item)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as ex:
                print(f"{Fore.YELLOW}Não foi possível remover {path}: {ex}{Style.RESET_ALL}")
    if os.path.exists("./wandb"):
        try:
            shutil.rmtree("./wandb")
        except Exception as ex:
            print(f"{Fore.YELLOW}Falha ao limpar wandb: {ex}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Limpeza concluída.{Style.RESET_ALL}")

limpar_ficheiros_temporarios()
