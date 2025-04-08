from transformers import MT5ForConditionalGeneration
from pre_processing import load_and_preprocess, tokenizer
from utils import normalize_text
import torch
import evaluate

# Carregar modelo treinado
model = MT5ForConditionalGeneration.from_pretrained("./modelo-final")

# Recarregar e tokenizar dataset (como no treino)
dataset = load_and_preprocess()
split_dataset = dataset.train_test_split(test_size=0.2)
eval_dataset = split_dataset["test"]

# Métricas BLEU e ROUGE
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# Gerar previsões e comparar com referências
preds = []
refs = []
bleu_preds = []
bleu_refs = []
label_coverage = []

for example in eval_dataset:
    input_ids = torch.tensor([example["input_ids"]])
    attention_mask = torch.tensor([example["attention_mask"]])
    labels = example["labels"]

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    ref = tokenizer.decode(labels, skip_special_tokens=True)

    preds.append(pred)
    refs.append(ref)
    bleu_preds.append(pred)
    bleu_refs.append([ref])

    rouge_metric.add(prediction=pred, reference=ref)

    # Métrica personalizada: cobertura de labels
    input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
    labels_in_input = input_text.replace("Gere uma frase descritiva sobre:", "").split(",")
    normalized_labels = [normalize_text(label.strip()) for label in labels_in_input]
    normalized_pred = normalize_text(pred)
    covered = all(label in normalized_pred for label in normalized_labels)
    label_coverage.append(covered)

# Calcular métricas
bleu_score = bleu_metric.compute(predictions=bleu_preds, references=bleu_refs)
rouge_score = rouge_metric.compute()
coverage_score = sum(label_coverage) / len(label_coverage)

# Mostrar exemplos e métricas
for i in range(10):
    input_example = tokenizer.decode(eval_dataset[i]["input_ids"], skip_special_tokens=True)
    labels_associadas = input_example.replace("Gere uma frase descritiva sobre:", "").strip()

    print(f"🏷️ Labels: {labels_associadas}")
    print(f"🎯 Real: {refs[i]}")
    print(f"🤖 Previsto: {preds[i]}")
    print("------")

print(f"\n📊 BLEU score médio: {bleu_score['bleu']:.4f}")
print(f"📊 ROUGE-L F1 score: {rouge_score['rougeL']:.4f}")
print(f"📊 Cobertura de labels no texto gerado: {coverage_score:.2%}")
