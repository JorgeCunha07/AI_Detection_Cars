import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from dataset import load_data
from vocab import SOS, PAD, EOS
from tqdm import tqdm
import os
import logging


def generate_greedy(model, src_tensor, output_vocab, device):
    model.eval()
    encoder_outputs, hidden, cell = model.encoder(src_tensor)
    input_token = torch.tensor([output_vocab.word2idx[SOS]], dtype=torch.long).to(device)

    output_sentence = []
    for _ in range(30):
        output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
        word_id = output.argmax(1).item()
        word = output_vocab.idx2word.get(word_id, "<unk>")
        if word in [SOS, EOS]:
            break
        output_sentence.append(word)
        input_token = torch.tensor([word_id], dtype=torch.long).to(device)

    return " ".join(output_sentence)


def train_model(
        data_path: str = "data/frases_com_labels.json",
        save_dir: str = "checkpoints",
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(filename="logs/train.log", level=logging.INFO)

    train_dataset, val_dataset = load_data(data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    encoder = Encoder(len(train_dataset.input_vocab), 256, 512)
    decoder = Decoder(len(train_dataset.output_vocab), 256, 512)
    model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.output_vocab.word2idx[PAD])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    alpha = 0.2  # Peso da penalização de cobertura

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_accum = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for src, tgt in loop:
            src, tgt = src.to(device), tgt.to(device)

            output = model(src, tgt)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt_flat = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt_flat)

            # -------- Penalização por falta de labels --------
            with torch.no_grad():
                generated_texts = [
                    generate_greedy(model, s.unsqueeze(0), train_dataset.output_vocab, device)
                    for s in src
                ]

            coverage_losses = []
            for i in range(len(src)):
                input_labels = train_dataset.input_vocab.decode(src[i].tolist()).split()
                prediction = generated_texts[i].split()
                coverage_hits = [label for label in input_labels if label in prediction]
                coverage_score = len(coverage_hits) / len(input_labels) if input_labels else 0.0
                coverage_losses.append(1.0 - coverage_score)

            coverage_loss = torch.tensor(coverage_losses, dtype=torch.float).mean().to(device)
            total_loss = loss + alpha * coverage_loss
            # -------------------------------------------------

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_value = total_loss.item()
            total_loss_accum += total_loss_value
            loop.set_postfix(loss=total_loss_value)

        avg_loss = total_loss_accum / len(train_loader)
        logging.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"{save_dir}/model_epoch{epoch}.pt")
        train_dataset.input_vocab.save(f"{save_dir}/input_vocab.pkl")
        train_dataset.output_vocab.save(f"{save_dir}/output_vocab.pkl")

    print(f"✅ Treino concluído. Modelos guardados em '{save_dir}'")
    return model, train_dataset.input_vocab, train_dataset.output_vocab


model, input_vocab, output_vocab = train_model(
    data_path="data/frases_com_labels.json",
    save_dir="checkpoints",
    epochs=50,
    batch_size=64,
    lr=0.001,
)
