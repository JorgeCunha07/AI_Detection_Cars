import os
import sys
import json
import torch
import math
import logging
import multiprocessing # Importar multiprocessing
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback, TrainerCallback
from datasets import Dataset
from pathlib import Path
from torch.nn import CrossEntropyLoss

# --- Definições de Funções Globais ---

# Função de tokenização com loss masking (versão robusta com procura por IDs)
def tokenize(batch, tokenizer, max_seq_length): # Passar tokenizer e max_length como args
    outputs = tokenizer(
        batch["text"],
        truncation=True,
        padding="longest",
        max_length=max_seq_length,
        return_overflowing_tokens=False
    )
    input_ids_batch = outputs["input_ids"]
    labels_batch = []
    assistant_marker_str = "Assistente:"
    assistant_marker_ids = tokenizer.encode(assistant_marker_str, add_special_tokens=False)
    if not assistant_marker_ids: # Segurança extra se o marcador for vazio ou só tokens especiais
         logger.warning(f"O marcador '{assistant_marker_str}' resultou em IDs vazios: {assistant_marker_ids}. O masking pode falhar.")

    for i, input_ids in enumerate(input_ids_batch):
        marker_pos = -1
        if assistant_marker_ids: # Só procurar se temos IDs válidos para o marcador
            for k in range(len(input_ids) - len(assistant_marker_ids) + 1):
                # Verificar se a fatia corresponde aos IDs do marcador
                if input_ids[k:k+len(assistant_marker_ids)] == assistant_marker_ids:
                    marker_pos = k
                    break # Encontrado

        prompt_length = 0
        if marker_pos != -1:
            prompt_length = marker_pos + len(assistant_marker_ids)
        else:
            prompt_length = len(input_ids) # Fallback: mascarar tudo
            # logger.warning(f"Aviso: Marcador IDs {assistant_marker_ids} não encontrado no exemplo {i}. Mascarando tudo.") # Log pode ser muito verboso

        example_labels = list(input_ids)
        for j in range(prompt_length):
            if j < len(example_labels):
                 example_labels[j] = -100
        labels_batch.append(example_labels)

    outputs["labels"] = labels_batch
    return outputs

# Função para calcular métricas de avaliação (perplexidade)
def compute_metrics(eval_pred):
    # Setup inicial do logger dentro da função para multiprocessing, se necessário
    # (Embora o Trainer geralmente lide bem com logs do processo principal)
    # logger = logging.getLogger(__name__) # Opcional

    logits, labels = eval_pred

    if isinstance(logits, tuple): logits = logits[0]
    if isinstance(labels, tuple): labels = labels[0]
    if not isinstance(logits, torch.Tensor): logits = torch.tensor(logits)
    if not isinstance(labels, torch.Tensor): labels = torch.tensor(labels)

    active_loss = labels != -100
    if not torch.any(active_loss):
        # logger.warning("Nenhum label ativo encontrado na avaliação.") # Pode ser verboso
        return {"perplexity": float('inf'), "loss": float('inf'), "accuracy": 0.0}

    active_logits = logits.view(-1, logits.size(-1))[active_loss.view(-1)]
    active_labels = labels.view(-1)[active_loss.view(-1)]

    loss = float('inf')
    perplexity = float('inf')
    nll_loss = float('inf')
    accuracy = 0.0

    try:
        loss_fct = CrossEntropyLoss(reduction="mean")
        loss = loss_fct(active_logits.to(active_labels.device), active_labels).item()
    except Exception as e:
        # logger.error(f"Erro ao calcular CrossEntropyLoss: {e}") # Verboso
        pass # Continua para calcular outras métricas se possível

    try:
      with torch.no_grad():
          log_probs = torch.nn.functional.log_softmax(active_logits, dim=-1)
          nll = -log_probs.gather(dim=-1, index=active_labels.unsqueeze(1)).squeeze(1)
          nll_loss = nll.mean().item()
          capped_loss = min(nll_loss, 10)
          perplexity = math.exp(capped_loss)
    except Exception as e:
       # logger.error(f"Erro ao calcular NLL/Perplexity: {e}") # Verboso
       pass

    try:
      predictions = torch.argmax(active_logits, dim=-1)
      correct = (predictions == active_labels).float().sum()
      total = active_labels.numel()
      if total > 0: accuracy = (correct / total).item()
    except Exception as e:
       # logger.error(f"Erro ao calcular Acurácia: {e}") # Verboso
       pass

    # logger.info(f"Métricas: Perp={perplexity:.2f}, Loss={loss:.4f}, NLL={nll_loss:.4f}, Acc={accuracy:.4f}") # Log feito pelo Trainer

    return {
        "perplexity": perplexity,
        "loss": loss,
        "accuracy": accuracy
    }

# Callback para gerar amostras no final de cada época
class SampleGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts, output_dir):
        # Não guardar referência ao modelo aqui para evitar problemas de serialização/multiprocessing
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        model = kwargs.get('model') # Obter o modelo atual do Trainer via kwargs
        if model is None:
            logger.warning("Não foi possível obter o modelo no SampleGenerationCallback.")
            return

        logger.info(f"Gerando amostras para a Época {int(epoch)}...")
        model.eval()
        outputs = []
        device = model.device # Obter o dispositivo do modelo

        for prompt in self.prompts:
            input_text = f"Usuário: {prompt}\nAssistente:"
            # Tokenizar dentro do callback
            encoding = self.tokenizer(input_text, return_tensors="pt", truncation=False) # Não truncar o prompt
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)


            # Calcular max_length para a geração (prompt + alguma margem)
            # Usar um valor fixo ou baseado no prompt é mais seguro que max_seq_length global aqui
            gen_max_length = input_ids.shape[1] + 100 # Exemplo: permitir gerar mais 100 tokens

            with torch.no_grad():
                output_sequences = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=gen_max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    no_repeat_ngram_size=2,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            outputs.append(f"--- Prompt: {prompt}\n--- Gerado:\n{generated_text}\n")

        sample_file = self.output_dir / f"samples_epoch_{int(epoch)}.txt"
        try:
          with open(sample_file, "w", encoding="utf-8") as f: f.write("\n".join(outputs))
          logger.info(f"Amostras geradas salvas em: {sample_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar amostras geradas: {e}")
        model.train() # Voltar ao modo de treino se aplicável


# --- Função Principal de Execução ---
def run_training():
    # Chamada OBRIGATÓRIA para multiprocessing em Windows
    multiprocessing.freeze_support()

    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("treinamento.log", encoding='utf-8', mode='a'), # Usar mode 'a' para append
            logging.StreamHandler()
        ]
    )
    global logger # Tornar logger global para uso nas funções se necessário (ou passar como arg)
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info("Iniciando Script de Treinamento...")
    logger.info("="*50)

    # Setup do Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo em uso: {device}")
    if not torch.cuda.is_available():
        logger.warning("CUDA não disponível, usando CPU. O treino será muito lento.")

    # Definição de Caminhos (com fallback para notebooks)
    try:
        script_dir = Path(__file__).resolve().parent
        base_dir = script_dir.parent.parent # Assumindo scr/treino/ficheiro.py -> vai para AI_Detection_Cars
        chat_dir = base_dir / "LNSCIA" / "ChatBot" / "scr" / "chat" # Ajustar conforme estrutura exata
        if not chat_dir.is_dir():
             # Tentar outra estrutura comum? Ou falhar?
             logger.warning(f"Diretório chat não encontrado em {chat_dir}. Tentando caminho relativo ao script...")
             chat_dir = script_dir / ".." / "chat" # Assume que 'chat' está um nível acima de 'treino'
             chat_dir = chat_dir.resolve() # Resolve '..'

        if not chat_dir.is_dir():
            logger.error(f"ERRO CRÍTICO: Diretório 'chat' não encontrado em {chat_dir} ou estruturas alternativas.")
            sys.exit(1)

    except NameError:
         logger.warning("__file__ não definido (provavelmente num notebook). Usando caminhos relativos ao diretório atual.")
         chat_dir = Path("./chat").resolve() # Assume subdiretório 'chat'
         if not chat_dir.is_dir():
              logger.error(f"ERRO CRÍTICO: Diretório 'chat' não encontrado no diretório atual.")
              sys.exit(1)

    modelo_output_dir = chat_dir / "gpt2-chat-finetuned"
    dados_json_path = chat_dir / "dialogos_validos4.json"
    trainer_output_dir = chat_dir / "trainer_output" # Diretório para checkpoints/logs do trainer
    logger.info(f"Diretório base 'chat': {chat_dir}")
    logger.info(f"Ficheiro de dados: {dados_json_path}")
    logger.info(f"Diretório de saída do modelo final: {modelo_output_dir}")
    logger.info(f"Diretório de saída do Trainer: {trainer_output_dir}")

    if not dados_json_path.exists():
        logger.error(f"ERRO: Ficheiro de dados JSON não encontrado em {dados_json_path}")
        sys.exit(1)

    # Carregar e filtrar dados
    logger.info(f"A carregar dados de {dados_json_path.name}...")
    try:
        with open(dados_json_path, encoding="utf-8") as f: data = json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar ou fazer parse do JSON de {dados_json_path}: {e}")
        sys.exit(1)

    seen = set()
    filtrados = []
    logger.info(f"Filtrando dados...")
    logger.info(f"Antes da limpeza, total de exemplos: {len(data)}")
    for idx, d in enumerate(data):
        if not isinstance(d, dict):
            logger.warning(f"Item {idx} não é um dicionário, ignorando: {d}")
            continue
        inp = d.get("input", "").strip()
        out = d.get("output", "").strip()
        k = (inp, out)
        if k not in seen and inp and out:
            seen.add(k)
            filtrados.append({"input": inp, "output": out})
    logger.info(f"Após limpeza, total de exemplos válidos e únicos: {len(filtrados)}")

    if not filtrados:
        logger.error("ERRO: Nenhum dado válido encontrado após a filtragem.")
        sys.exit(1)

    # Preparar textos
    texts = [f"Usuário: {d['input']}\nAssistente: {d['output']}" for d in filtrados]

    # Carregar Tokenizador e Modelo Base
    model_name = "pierreguillou/gpt2-small-portuguese"
    logger.info(f"A carregar tokenizador {model_name}...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Erro ao carregar o tokenizador {model_name}: {e}")
        sys.exit(1)

    # Analisar comprimentos e definir max_seq_length
    logger.info("Analisando comprimentos dos textos...")
    lengths = [len(tokenizer.encode(text, max_length=tokenizer.model_max_length + 10, truncation=False)) for text in texts] # Verificar sem truncar
    avg_length = sum(lengths) / len(lengths)
    max_length_data = max(lengths)
    logger.info(f"Comprimento médio: {avg_length:.2f}, Máximo nos dados: {max_length_data}")

    # Definir max_seq_length (ex: min(512, model_max_length))
    max_seq_length = min(512, tokenizer.model_max_length)
    logger.info(f"Comprimento máximo definido para tokenização (max_seq_length): {max_seq_length}")
    if max_length_data > max_seq_length:
         logger.warning(f"Existem exemplos ({sum(l > max_seq_length for l in lengths)}) mais longos que max_seq_length ({max_seq_length}). Serão truncados.")


    logger.info(f"A carregar modelo base {model_name} para o dispositivo {device}...")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo {model_name}: {e}")
        sys.exit(1)

    # Criar dataset e tokenizar
    logger.info("A criar e tokenizar os datasets...")
    dataset = Dataset.from_dict({"text": texts})
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # Usar multiprocessing para mapear
    num_proc = 1
    logger.warning(f"Forçando num_proc={num_proc} para mapeamento da tokenização para evitar erros de serialização.")

    # Usar functools.partial para passar args extras para tokenize dentro do .map
    from functools import partial
    tokenize_with_args = partial(tokenize, tokenizer=tokenizer, max_seq_length=max_seq_length)

    try:
        train_dataset = split_dataset["train"].map(
            tokenize_with_args, batched=True, num_proc=num_proc, remove_columns=["text"]
        )
        val_dataset = split_dataset["test"].map(
            tokenize_with_args, batched=True, num_proc=num_proc, remove_columns=["text"]
        )
    except Exception as e:
        logger.error(f"Erro durante a tokenização com multiprocessing: {e}")
        logger.info("Tentando tokenizar sem multiprocessing (num_proc=1)...")
        num_proc = 1
        try:
            train_dataset = split_dataset["train"].map(
                tokenize_with_args, batched=True, num_proc=num_proc, remove_columns=["text"]
            )
            val_dataset = split_dataset["test"].map(
                tokenize_with_args, batched=True, num_proc=num_proc, remove_columns=["text"]
            )
        except Exception as e2:
             logger.error(f"Erro fatal durante a tokenização (mesmo sem multiprocessing): {e2}")
             sys.exit(1)


    logger.info("Tokenização concluída.")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("ERRO: Um dos datasets ficou vazio após tokenização.")
        sys.exit(1)
    logger.info(f"Tamanho do dataset de treino: {len(train_dataset)}")
    logger.info(f"Tamanho do dataset de validação: {len(val_dataset)}")


    # Configuração dos Argumentos de Treino
    args = TrainingArguments(
        output_dir=str(trainer_output_dir),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1, # MANTIDO EM 1
        gradient_accumulation_steps=2,
        gradient_checkpointing=True, # MANTIDO
        num_train_epochs=8, # MANTIDO
        warmup_ratio=0.1,
        learning_rate=3e-5, # MANTIDO
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=torch.cuda.is_available(), # Ativar FP16 apenas se houver CUDA
        # report_to="tensorboard", # Usar default (tensorboard se instalado)
        remove_unused_columns=True, # Colunas não usadas são removidas
        # seed=42 # Adicionar seed para reprodutibilidade do treino
    )

    # Collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Callbacks
    # Passar apenas tokenizer para o callback
    sample_callback = SampleGenerationCallback(
        tokenizer=tokenizer,
        prompts=[ # Prompts de teste
            "Qual é a velocidade máxima em autoestrada?",
            "O que devo fazer num cruzamento com prioridade à direita?",
            "Como posso verificar o nível do óleo do motor?",
            "Quais são os documentos obrigatórios para circular?"
        ],
        output_dir=str(trainer_output_dir / "gerados")
    )
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01),
        sample_callback
    ]

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    # Treino e Avaliação
    train_result = None
    final_metrics = None
    try:
        logger.info("Iniciando o ciclo de treino...")
        train_result = trainer.train()

        logger.info("Treino concluído. Salvando métricas de treino...")
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        logger.info("Salvando o melhor modelo encontrado durante o treino...")
        modelo_output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(modelo_output_dir)
        tokenizer.save_pretrained(modelo_output_dir) # Salvar tokenizer junto com o modelo
        logger.info(f"Melhor modelo e tokenizer salvos em {modelo_output_dir}")

        logger.info("Avaliando o melhor modelo no conjunto de validação...")
        final_metrics = trainer.evaluate(eval_dataset=val_dataset) # Avaliar no val_dataset
        logger.info(f"Métricas finais de avaliação: {final_metrics}")
        trainer.log_metrics("eval_final", final_metrics)
        trainer.save_metrics("eval_final", final_metrics)

    except Exception as e:
         logger.exception(f"Ocorreu um erro durante o treino ou avaliação: {e}")
         logger.warning("Tentando salvar o estado atual do Trainer...")
         try:
             trainer.save_state()
             trainer.save_model(str(trainer_output_dir / "checkpoint-interrupted"))
             logger.warning(f"Estado e modelo (interrompido) salvos em {trainer_output_dir}")
         except Exception as e_save:
             logger.error(f"Não foi possível salvar o estado/modelo após erro: {e_save}")

    # Teste final com o modelo carregado (se o treino correu minimamente bem)
    if train_result is not None: # Verificar se o treino iniciou
        logger.info("Testando o modelo final (carregado pelo Trainer ou último checkpoint) com exemplos...")
        # O modelo no trainer já é o melhor se load_best_model_at_end=True e o treino terminou
        model_to_test = trainer.model
        model_to_test.eval()
        model_to_test.to(device)

        test_prompts = [ # Usar os mesmos prompts do callback
            "Qual é a velocidade máxima em autoestrada?",
            "O que devo fazer num cruzamento com prioridade à direita?",
            "Como posso verificar o nível do óleo do motor?",
            "Quais são os documentos obrigatórios para circular?"
        ]

        for prompt in test_prompts:
            input_text = f"Usuário: {prompt}\nAssistente:"
            encoding = tokenizer(input_text, return_tensors="pt", truncation=False)
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            gen_max_length = input_ids.shape[1] + 100

            with torch.no_grad():
                output = model_to_test.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=gen_max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    no_repeat_ngram_size=2,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            logger.info(f"\n--- Prompt: {prompt}\n--- Resposta gerada:\n{generated_text}\n")

    logger.info("="*50)
    logger.info("Script de Treinamento Concluído.")
    logger.info("="*50)


# --- Ponto de Entrada Principal do Script ---
if __name__ == '__main__':
    run_training() # Chama a função que contém toda a lógica