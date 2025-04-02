import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import json
import os
import shutil
import wandb
from colorama import Fore, Style, init

# Inicializa o colorama com autoreset
init(autoreset=True)

# Verifica se a GPU está disponível; se não, aborta
if not torch.cuda.is_available():
    print(f"{Fore.RED}ERRO: GPU não encontrada! Este script requer uma GPU para treino. Abortando.{Style.RESET_ALL}")
    exit(1)

device = torch.device("cuda")
print(f"{Fore.GREEN}Usando GPU: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")

# Modelo adaptado para o português
model_name = "pierreguillou/gpt2-small-portuguese"
print(f"{Fore.CYAN}Carregando modelo base {model_name}...{Style.RESET_ALL}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Carrega em fp16; o treino será feito em FP32 para evitar problemas
    device_map="auto"
)

# Se o pad_token não estiver definido ou for igual ao eos_token, adicione um novo token de pad.
if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
print(f"{Fore.YELLOW}Pad token configurado como: {tokenizer.pad_token_id}{Style.RESET_ALL}")

def load_training_data(codigo_file, questions_file):
    print(f"{Fore.CYAN}Carregando dados de treino...{Style.RESET_ALL}")
    with open(codigo_file, encoding="utf-8") as f:
        codigo_data = json.load(f)
    with open(questions_file, encoding="utf-8") as f:
        questions_data = json.load(f)
    
    examples = []
    # Cria exemplos a partir dos artigos do Código da Estrada
    for article in codigo_data.get("articles", []):
        prompt = f"Explique o seguinte artigo do Código da Estrada: {article['article']}\nResposta: {article['text']}"
        examples.append({"text": prompt})
    # Cria exemplos a partir das perguntas e respostas
    for item in questions_data:
        for resposta in item.get("respostas_corretas", []):
            prompt = f"Pergunta: {item['pergunta']}\nResposta: {resposta}"
            examples.append({"text": prompt})
    print(f"{Fore.GREEN}Dados carregados: {len(examples)} exemplos{Style.RESET_ALL}")
    return examples

# Ajuste os caminhos conforme necessário
train_examples = load_training_data("../../Documentacao/BomCondutor/Codigo_Estrada.json", "../../Documentacao/Questoes/questoes.json")
dataset = Dataset.from_list(train_examples)

# Divide o dataset em 90% treino e 10% avaliação
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Tokeniza os dados usando max_length=512
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator para modelagem de linguagem causal (MLM desativado)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Configuração dos parâmetros de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=50,
    save_steps=100,
    save_total_limit=2,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    fp16=False,
    gradient_checkpointing=False,
    report_to="wandb",
    max_grad_norm=1.0
)

# (Opcional) compute_metrics se necessário
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"loss": float(torch.tensor(logits).mean().item())}

wandb.init(project="codigo-estrada-chatbot", reinit=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

try:
    print(f"{Fore.GREEN}Iniciando treino...{Style.RESET_ALL}")
    trainer.train()
    print(f"{Fore.CYAN}Salvando modelo treinado...{Style.RESET_ALL}")
    trainer.save_model("./trained_model")
    tokenizer.save_pretrained("./trained_model")
except Exception as e:
    print(f"{Fore.RED}Erro durante o treino: {e}{Style.RESET_ALL}")
    raise e
finally:
    torch.cuda.empty_cache()
    wandb.finish()

def limpar_ficheiros_temporarios():
    print(f"{Fore.CYAN}Limpando ficheiros temporários...{Style.RESET_ALL}")
    if os.path.exists("./trained_model"):
        for item in os.listdir("./trained_model"):
            if item.startswith("checkpoint-") or item == "runs":
                path = os.path.join("./trained_model", item)
                try:
                    shutil.rmtree(path)
                except Exception as ex:
                    print(f"{Fore.YELLOW}Aviso: Não foi possível remover {path}: {ex}{Style.RESET_ALL}")
    if os.path.exists("./wandb"):
        try:
            shutil.rmtree("./wandb")
        except Exception as ex:
            print(f"{Fore.YELLOW}Aviso: Não foi possível remover a pasta wandb: {ex}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Ficheiros temporários limpos com sucesso!{Style.RESET_ALL}")

try:
    limpar_ficheiros_temporarios()
except Exception as ex:
    print(f"{Fore.YELLOW}Erro durante a limpeza: {ex}{Style.RESET_ALL}")
