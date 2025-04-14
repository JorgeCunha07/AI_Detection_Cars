import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from dataset import load_data
from vocab import SOS, EOS, PAD, UNK
from tqdm import tqdm
import os
import logging
import string


def generate_greedy(model, src_tensor, output_vocab, device, max_len=50):
    was_training = model.training
    model.encoder.eval()
    model.decoder.eval()

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        input_token = torch.tensor([output_vocab.word2idx[SOS]], dtype=torch.long).to(device)

        output_sentence = []
        eos_count = 0

        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
            word_id = output.argmax(1).item()
            word = output_vocab.idx2word.get(word_id, UNK)

            if word == EOS:
                eos_count += 1
                if eos_count >= 1:
                    break
            elif word != SOS:
                output_sentence.append(word)

            input_token = torch.tensor([word_id], dtype=torch.long).to(device)

    model.train(was_training)
    return " ".join(output_sentence)

# Remove pontuacao do texto
def clean(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

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

    encoder = Encoder(len(train_dataset.input_vocab), 256, 512)
    decoder = Decoder(len(train_dataset.output_vocab), 256, 512)
    model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.output_vocab.word2idx[PAD])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_accum = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        alpha = min(1.0, epoch * 0.1)  # aumenta progressivamente

        for src, tgt in loop:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt_flat = tgt[:, 1:].reshape(-1)
            ce_loss = criterion(output, tgt_flat)

            # Geração de frases para penalização
            with torch.no_grad():
                generated_texts = [
                    generate_greedy(model, s.unsqueeze(0), train_dataset.output_vocab, device,
                                    max_len=20 + s.size(0) * 5)
                    for s in src
                ]

            coverage_losses = []
            for i in range(len(src)):
                input_labels = train_dataset.input_vocab.decode(src[i].tolist()).split()
                
                input_labels = [label.replace("_", " ") for label in input_labels]

                prediction = generated_texts[i]

                cleaned_prediction = clean(prediction)
                coverage_hits = [label for label in input_labels if label.lower() in cleaned_prediction]
                
                score = len(coverage_hits) / len(input_labels) if input_labels else 0.0
                focal_loss = (1.0 - score) ** 2
                coverage_losses.append(focal_loss)

            coverage_loss = torch.tensor(coverage_losses, dtype=torch.float).mean().to(device)
            total_loss = ce_loss + alpha * coverage_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_accum += total_loss.item()
            loop.set_postfix(loss=total_loss.item(), alpha=alpha)

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
    batch_size=16,
    lr=0.0003,
)
