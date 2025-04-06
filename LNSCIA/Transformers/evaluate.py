from transformers import MT5ForConditionalGeneration
from pre_processing import load_and_preprocess, tokenizer
from datasets import load_dataset

# Carregar modelo treinado
model = MT5ForConditionalGeneration.from_pretrained("./modelo-final")

# Recarregar e tokenizar dataset (mesmo que no treino)
dataset = load_and_preprocess("./data/frases_com_labels.json")
split_dataset = dataset.train_test_split(test_size=0.2)
eval_dataset = split_dataset["test"]

# Gerar previsões
outputs = model.generate(
    input_ids=[x["input_ids"] for x in eval_dataset],
    attention_mask=[x["attention_mask"] for x in eval_dataset],
    max_length=64,
    num_beams=4,
)

preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
refs = tokenizer.batch_decode(eval_dataset["labels"], skip_special_tokens=True)

# Mostrar amostras
for i in range(5):
    print(f"🎯 Real: {refs[i]}")
    print(f"🤖 Previsto: {preds[i]}")
    print("------")
