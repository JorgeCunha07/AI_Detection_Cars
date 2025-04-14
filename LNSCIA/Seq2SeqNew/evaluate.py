import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from dataset import load_data
from vocab import Vocab, SOS, EOS, PAD, UNK
import os
import csv
import string


def evaluate_model(
        data_path: str,
        model_path: str,
        input_vocab_path: str,
        output_vocab_path: str,
        result_csv_path: str = "results/eval_results.csv",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)

    # Carregar vocabulários
    input_vocab = Vocab.load(input_vocab_path)
    output_vocab = Vocab.load(output_vocab_path)

    # Preparar modelo
    encoder = Encoder(len(input_vocab), 256, 512)
    decoder = Decoder(len(output_vocab), 256, 512)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dataset de validação
    _, val_dataset = load_data(data_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=val_dataset.collate_fn)

    smoothie = SmoothingFunction().method4
    results = []

    # Remove pontuacao do texto
    def clean(text):
        return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

    def generate(input_tensor):
        encoder_outputs, hidden, cell = model.encoder(input_tensor)
        input_token = torch.tensor([output_vocab.word2idx[SOS]], dtype=torch.long).to(device)
        output_sentence = []

        for _ in range(30):
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1).item()
            word = output_vocab.idx2word.get(top1, UNK)
            if word in [SOS, EOS]:
                break
            output_sentence.append(word)
            input_token = torch.tensor([top1], dtype=torch.long).to(device)

        return " ".join(output_sentence)

    for src, tgt in val_loader:
        src = src.to(device)
        pred_text = generate(src[0].unsqueeze(0))

        # Limpar frase target
        label_tokens = [
            output_vocab.idx2word[i]
            for i in tgt[0].tolist()
            if i not in [output_vocab.word2idx[SOS], output_vocab.word2idx[EOS], output_vocab.word2idx[PAD]]
        ]
        label_clean = " ".join(label_tokens)

        # BLEU
        ref_tokens = label_clean.split()
        pred_tokens = pred_text.split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)

        # Cobertura de labels
        input_text = val_dataset.input_vocab.decode(src[0].tolist())

        input_text_formatted = input_text.replace("_", " ")
        input_labels = input_text_formatted.split()

        cleaned_pred_tokens = [clean(token) for token in pred_tokens]
        coverage_hits = [label for label in input_labels if label.lower() in cleaned_pred_tokens]
        label_coverage = len(coverage_hits) / len(input_labels) if input_labels else 0.0

        results.append({
            "labels": input_text,
            "target": label_clean,
            "prediction": pred_text,
            "bleu": round(bleu, 4),
            "label_coverage": round(label_coverage, 2)
        })

    # Ordenar os resultados do melhor para o pior com base na cobertura e BLEU
    sorted_results = sorted(results, key=lambda x: (x["label_coverage"], x["bleu"]), reverse=True)

    # Guardar CSV ordenado
    with open(result_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted_results[0].keys())
        writer.writeheader()
        writer.writerows(sorted_results)

    print("📄 Resultados ordenados guardados em results/eval_results_sorted.csv")

    # Imprimir médias
    avg_bleu = sum(r["bleu"] for r in results) / len(results)
    avg_coverage = sum(r["label_coverage"] for r in results) / len(results)

    print(f"✅ Avaliação concluída. Resultados guardados em {result_csv_path}")
    print(f"📊 Média BLEU: {avg_bleu:.4f}")
    print(f"📊 Média label coverage: {avg_coverage:.4f}")

    return results


evaluate_model(
    data_path="data/frases_com_labels.json",
    model_path="checkpoints/model_epoch50.pt",
    input_vocab_path="checkpoints/input_vocab.pkl",
    output_vocab_path="checkpoints/output_vocab.pkl",
    result_csv_path="results/eval_results.csv"
)
