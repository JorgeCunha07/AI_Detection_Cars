import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def main():
    print("Carregando o dataset 'marquesafonso/wikipedia-pt-embeddings'...")
    # Carregar o dataset a partir do Hugging Face Datasets
    dataset = load_dataset("marquesafonso/wikipedia-pt-embeddings")
    
    # Se o dataset não possuir um split "train", utiliza o split "default" e divide manualmente
    if "train" not in dataset:
        dataset = dataset["default"].train_test_split(test_size=0.1, seed=42)
    else:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    
    # Nome do modelo pré-treinado (substitua se desejar outro)
    model_name = "pierreguillou/gpt2-small-portuguese"
    
    # Carregar o tokenizer e o modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Função de tokenização: combina "title" e "chunk" para gerar o texto de entrada
    def tokenize_function(examples):
        texts = [f"{title}: {chunk}" for title, chunk in zip(examples["title"], examples["chunk"])]
        return tokenizer(texts, truncation=True, max_length=512)
    
    # Remove todas as colunas originais (para manter apenas as colunas geradas pelo tokenizer)
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Criar o data collator para treinamento (mlm=False pois é para modelo de geração)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Configuração dos argumentos de treinamento
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned-wikipedia-pt",
        overwrite_output_dir=True,
        num_train_epochs=3,                # Ajuste conforme necessário
        per_device_train_batch_size=2,     # Ajuste conforme a memória disponível
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,  # Utilize fp16 se a sua GPU suportar
    )
    
    # Inicializar o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )
    
    # Iniciar o treinamento
    trainer.train()

if __name__ == "__main__":
    main()
