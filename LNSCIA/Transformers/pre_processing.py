from datasets import load_dataset
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("google/mt5-small", legacy=False)


def preprocess(example):
    input_text = "Gere uma frase que descreva a seguinte cena: " + example["labels"]
    target_text = example["description"]

    # Tokenizar input com truncation e padding
    model_inputs = tokenizer(
        input_text, max_length=128, truncation=True, padding=False
    )

    # Tokenizar output (labels) com truncation e padding
    labels = tokenizer(
        target_text, max_length=128, truncation=True, padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_and_preprocess(path="./data/frases_com_labels.json"):
    dataset = load_dataset("json", data_files=path, split="train")
    tokenized = dataset.map(preprocess).shuffle(seed=42)
    return tokenized
