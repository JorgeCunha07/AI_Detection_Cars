import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import json
import os
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import wandb
from colorama import Fore, Style, init

# Inicializa o colorama
init()

# Verificar disponibilidade de GPU
if not torch.cuda.is_available():
    print(f"{Fore.RED}ERRO: GPU não encontrada! Este script requer uma GPU para treino.{Style.RESET_ALL}")
    print("Por favor, certifique-se de que:")
    print("1. Tem uma GPU NVIDIA instalada")
    print("2. Tem os drivers CUDA instalados")
    print("3. Tem o PyTorch com suporte CUDA instalado")
    exit(1)

# Configurar device para GPU
device = torch.device("cuda")
print(f"{Fore.GREEN}Usando GPU: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")

# Carrega o tokenizer e o modelo base
model_name = "microsoft/phi-2"
print(f"{Fore.CYAN}Carregando modelo base {model_name}...{Style.RESET_ALL}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Usa precisão reduzida para melhor performance
    device_map="auto"  # Gerencia automaticamente o carregamento na GPU
)

# Configurar o tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    print(f"{Fore.YELLOW}Configurado pad_token como eos_token{Style.RESET_ALL}")

# Função para carregar e preparar os dados
def load_and_prepare_data():
    print(f"{Fore.CYAN}Carregando dados de treino...{Style.RESET_ALL}")
    
    # Carrega os dados do Código da Estrada
    with open("../../Documentacao/BomCondutor/Codigo_Estrada.json", encoding="utf-8") as f:
        codigo_data = json.load(f)
    
    # Carrega as questões do quiz
    with open("../../Documentacao/Questoes/questoes.json", encoding="utf-8") as f:
        quiz_data = json.load(f)
    
    # Prepara os dados de treino
    training_data = []
    
    # Adiciona artigos do Código da Estrada
    for artigo in codigo_data.get("articles", []):
        training_data.append({
            "text": f"Artigo {artigo['article']}: {artigo['text']}",
            "reference": artigo.get("reference", "")
        })
    
    # Adiciona questões do quiz
    for questao in quiz_data:
        training_data.append({
            "text": f"Pergunta: {questao['pergunta']}\nResposta: {questao['respostas_corretas'][0]}",
            "reference": questao.get("referencia", "")
        })
    
    # Tokeniza os textos
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    # Cria o dataset
    dataset = Dataset.from_dict({
        "text": [item["text"] for item in training_data],
        "reference": [item["reference"] for item in training_data]
    })
    
    # Tokeniza o dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"{Fore.GREEN}Dados preparados: {len(training_data)} exemplos{Style.RESET_ALL}")
    return tokenized_dataset

# Configuração do treino
def setup_training():
    print(f"{Fore.CYAN}Configurando treino...{Style.RESET_ALL}")
    
    # Argumentos de treino
    training_args = TrainingArguments(
        output_dir="./trained_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,  # Usa precisão reduzida para melhor performance
        gradient_checkpointing=True,  # Economiza memória
        optim="adamw_torch",
        report_to="wandb"  # Integração com Weights & Biases
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    return training_args, data_collator

# Função principal de treino
def main():
    # Inicializa o Weights & Biases
    wandb.init(project="codigo-estrada-chatbot")
    
    # Carrega e prepara os dados
    dataset = load_and_prepare_data()
    
    # Configura o treino
    training_args, data_collator = setup_training()
    
    # Inicializa o trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: {"loss": pred.predictions.mean()}
    )
    
    # Inicia o treino
    print(f"{Fore.GREEN}Iniciando treino...{Style.RESET_ALL}")
    trainer.train()
    
    # Salva o modelo treinado
    print(f"{Fore.CYAN}Salvando modelo treinado...{Style.RESET_ALL}")
    trainer.save_model("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    
    print(f"{Fore.GREEN}Treino concluído! Modelo salvo em ./trained_model{Style.RESET_ALL}")
    
    # Limpa a memória da GPU
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
