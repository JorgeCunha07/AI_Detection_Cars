import os
import sys
import json
import torch
import math
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback, TrainerCallback
from datasets import Dataset
from pathlib import Path
from torch.nn import CrossEntropyLoss

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("treinamento.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Preparando o treino com melhorias e Early Stopping...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Dispositivo em uso: {device}")

# Diretórios e caminhos
chat_dir = Path(__file__).resolve().parent.parent / "chat"
modelo_output_dir = chat_dir / "gpt2-chat-finetuned"
dados_json_path = chat_dir / "dialogos_validos3.json"

# Carregar e filtrar dados
with open(dados_json_path, encoding="utf-8") as f:
    data = json.load(f)

seen = set()
filtrados = []
logger.info(f"Filtrando dados do ficheiro {dados_json_path}")
logger.info(f"Antes da limpeza, total de exemplos: {len(data)}")
for d in data:
    inp = d.get("input", "").strip()
    out = d.get("output", "").strip()
    k = (inp, out)
    if k not in seen and inp and out:
        seen.add(k)
        filtrados.append({"input": inp, "output": out})
    else:
        logger.warning(f"Exemplo duplicado ou inválido encontrado: {k}")
logger.info(f"Após limpeza, total de exemplos: {len(filtrados)}")

# Preparar os textos com marcadores explícitos
texts = [
    f"Usuário: {d['input']}\nAssistente: {d['output']}"
    for d in filtrados
]

# Analisar comprimentos para definir max_length ideal
model_name = "pierreguillou/gpt2-small-portuguese"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
lengths = [len(tokenizer.encode(text)) for text in texts]
avg_length = sum(lengths) / len(lengths)
max_length = max(lengths)
logger.info(f"Comprimento médio: {avg_length:.2f}, Máximo: {max_length}")
max_seq_length = min(512, max(256, int(avg_length * 1.5)))
logger.info(f"Comprimento máximo definido para tokenização: {max_seq_length}")

# Criar dataset a partir dos textos
dataset = Dataset.from_dict({"text": texts})

# Inicializar tokenizador e modelo base
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Função de tokenização com loss masking para os tokens do prompt
def tokenize(batch):
    outputs = tokenizer(batch["text"], truncation=True, padding="longest", max_length=max_seq_length)
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
            logger.warning(f"Aviso: marcador '{assistant_marker.strip()}' não encontrado no exemplo: {decoded[:50]}...")
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
    loss_fct = CrossEntropyLoss(reduction="mean")
    loss = loss_fct(active_logits, active_labels)
    
    # Usar o NLL Loss diretamente para calcular perplexidade mais precisa
    with torch.no_grad():
        # Transformar logits em probabilidades
        log_probs = torch.nn.functional.log_softmax(active_logits, dim=-1)
        # Selecionar log probabilidade dos tokens alvo
        nll = -log_probs.gather(dim=-1, index=active_labels.unsqueeze(1)).squeeze(1)
        # Média da negative log likelihood
        nll_loss = nll.mean().item()
        # Calcular perplexidade com limitação
        capped_loss = min(nll_loss, 10)
        perplexity = math.exp(capped_loss)
    
    # Calcular acurácia de tokens
    predictions = torch.argmax(active_logits, dim=-1)
    correct = (predictions == active_labels).float().sum()
    total = active_labels.numel()
    accuracy = (correct / total).item()
    
    logger.info(f"Perplexidade calculada: {perplexity:.2f} (NLL: {nll_loss:.4f}, Acurácia: {accuracy:.4f})")
    
    return {
        "perplexity": perplexity,
        "loss": loss.item(),
        "accuracy": accuracy
    }

# Callback para gerar amostras no final de cada época
class SampleGenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, prompts, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        model = kwargs.get('model', self.model)
        
        # Gerar algumas respostas para prompts de teste
        outputs = []
        for prompt in self.prompts:
            input_ids = self.tokenizer.encode(f"Usuário: {prompt}\nAssistente:", return_tensors="pt").to(model.device)
            # Criar a attention_mask com valor 1 para cada token
            attention_mask = torch.ones_like(input_ids)
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                no_repeat_ngram_size=2
            )
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(f"Prompt: {prompt}\nGerado: {generated_text}\n")
            
        # Salvar em arquivo
        with open(f"{self.output_dir}/samples_epoch_{int(epoch)}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(outputs))
        
        logger.info(f"Amostras geradas salvas para época {int(epoch)}")

# Exemplos de prompts para testar
test_prompts = [
    "Qual é a velocidade máxima em autoestrada?",
    "O que devo fazer num cruzamento com prioridade à direita?"
]

# Criar diretório para amostras geradas
Path("gerados").mkdir(exist_ok=True)

# Configuração dos argumentos de treino com early stopping e avaliação com perplexidade
args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    evaluation_strategy="steps",            # Avaliação em passos específicos
    eval_steps=500,                         # Avaliar a cada 500 passos
    per_device_train_batch_size=4,          # Tamanho do batch de treino
    per_device_eval_batch_size=4,           # Tamanho do batch de validação
    gradient_accumulation_steps=2,          # Acumulação de gradientes para simular batch maior
    num_train_epochs=20,                    # Número máximo de épocas (o treino pode parar antes)
    warmup_ratio=0.1,                       # Aquecimento de 10% dos passos totais
    logging_steps=100,                      # Passos para logar informações
    save_strategy="steps",                  # Salvar em passos específicos
    save_steps=500,                         # Salvar a cada 500 passos
    save_total_limit=3,                     # Manter apenas os 3 melhores checkpoints
    learning_rate=5e-5,                     # Learning rate inicial
    lr_scheduler_type="cosine",             # Scheduler cosine com decay
    weight_decay=0.01,                      # Regularização L2
    fp16=True if torch.cuda.is_available() else False,  # Precisão mista (se disponível)
    report_to="none",                       # Não reportar para serviços externos
    remove_unused_columns=False,
    load_best_model_at_end=True,            # Carrega o melhor modelo ao final
    metric_for_best_model="perplexity",
    greater_is_better=False                 # Menor perplexidade é melhor
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Instanciar o callback para geração de amostras
sample_callback = SampleGenerationCallback(
    model=model, 
    tokenizer=tokenizer, 
    prompts=test_prompts,
    output_dir="gerados"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        sample_callback
    ]
)

logger.info("Iniciando treino com Early Stopping...")
trainer.train()

logger.info("Salvando modelo...")
modelo_output_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(modelo_output_dir)
tokenizer.save_pretrained(modelo_output_dir)
logger.info(f"Modelo salvo em {modelo_output_dir}")

# Testar o modelo final com alguns exemplos
logger.info("Testando modelo final...")
model.eval()
for prompt in test_prompts:
    input_text = f"Usuário: {prompt}\nAssistente:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    # Criar a attention_mask para o input
    attention_mask = torch.ones_like(input_ids)
    # Gerar resposta
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        no_repeat_ngram_size=2
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info(f"\nPrompt: {prompt}\nResposta gerada: {generated_text}\n")

logger.info("Treinamento concluído com sucesso!")
