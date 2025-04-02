# train.py — treino completo de modelo Seq2Seq com Attention, métricas e checkpoints

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
import time

# ============================
# Parâmetros configuráveis
# ============================
BATCH_SIZE = 128
NUM_EPOCHS = 80
EMB_DIM = 300
ENC_HIDDEN_DIM = 256
DEC_HIDDEN_DIM = 512
LEARNING_RATE = 0.001
DROPOUT = 0.3
TEACHER_FORCING_RATIO = 0.5
CLIP = 1
MODEL_DIR = './models/'

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# ============================
# Dataset personalizado
# ============================
class Seq2SeqDataset(Dataset):
    def __init__(self, encoder_data, decoder_input, decoder_target):
        self.encoder_data = encoder_data
        self.decoder_input = decoder_input
        self.decoder_target = decoder_target

    def __len__(self):
        return len(self.encoder_data)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.encoder_data[idx]),
            torch.LongTensor(self.decoder_input[idx]),
            torch.LongTensor(self.decoder_target[idx])
        )

# ============================
# Modelo com Attention (Luong)
# ============================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[0:1], hidden[1:2]), dim=2)
        cell = torch.cat((cell[0:1], cell[1:2]), dim=2)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim*2 + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(enc_hidden_dim*2 + emb_dim, dec_hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(enc_hidden_dim*2 + dec_hidden_dim + emb_dim, output_dim)
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden.permute(1, 0, 2), encoder_outputs)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        lstm_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2)).squeeze(1)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)

        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

# ============================
# Ciclo de treino + avaliação
# ============================
def train(model, iterator, optimizer, criterion, clip=1):
    model.train()
    epoch_loss = 0

    for src, trg_in, trg_out in tqdm(iterator):
        src, trg_in, trg_out = src.to(device), trg_in.to(device), trg_out.to(device)
        optimizer.zero_grad()
        output = model(src, trg_in, TEACHER_FORCING_RATIO)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg_out = trg_out[:, 1:].reshape(-1)
        loss = criterion(output, trg_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg_in, trg_out in iterator:
            src, trg_in, trg_out = src.to(device), trg_in.to(device), trg_out.to(device)
            output = model(src, trg_in, 0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg_out = trg_out[:, 1:].reshape(-1)
            loss = criterion(output, trg_out)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# ============================
# Execução principal
# ============================
if __name__ == '__main__':
    with open(os.path.join(MODEL_DIR, 'label_tokenizer.pkl'), 'rb') as f:
        label_tokenizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'description_tokenizer.pkl'), 'rb') as f:
        description_tokenizer = pickle.load(f)

    label_train = np.load(os.path.join(MODEL_DIR, 'label_train.npy'))
    desc_train = np.load(os.path.join(MODEL_DIR, 'desc_train.npy'))

    decoder_input_data = desc_train[:, :-1]
    decoder_target_data = desc_train[:, 1:]

    dataset = Seq2SeqDataset(label_train, decoder_input_data, decoder_target_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    label_vocab_size = len(label_tokenizer.word_index) + 1
    desc_vocab_size = len(description_tokenizer.word_index) + 1

    enc = Encoder(label_vocab_size, emb_dim=EMB_DIM, hidden_dim=ENC_HIDDEN_DIM, dropout=DROPOUT).to(device)
    dec = Decoder(desc_vocab_size, emb_dim=EMB_DIM, enc_hidden_dim=ENC_HIDDEN_DIM, dec_hidden_dim=DEC_HIDDEN_DIM, dropout=DROPOUT).to(device)
    model = Seq2Seq(enc, dec).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        train_loss = train(model, train_loader, optimizer, criterion, clip=CLIP)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))
            print("Melhor modelo salvo!")
