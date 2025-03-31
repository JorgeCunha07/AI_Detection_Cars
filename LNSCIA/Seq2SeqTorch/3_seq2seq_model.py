import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
#from nltk.translate.bleu_score import sentence_bleu
#from rouge import Rouge
import random
import time  # Importa o módulo de tempo

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# Carregar os tokenizadores
with open('./models/label_tokenizer.pkl', 'rb') as f:
    label_tokenizer = pickle.load(f)
with open('./models/description_tokenizer.pkl', 'rb') as f:
    description_tokenizer = pickle.load(f)

# Carregar os dados pré-processados para treino
label_train = np.load('./models/label_train.npy')
desc_train = np.load('./models/desc_train.npy')

# Preparar os dados para o decoder (entrada e target deslocados)
decoder_input_data = desc_train[:, :-1]
decoder_target_data = desc_train[:, 1:]

# Definir um Dataset personalizado
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

# Parâmetros
batch_size = 128
num_epochs = 100
teacher_forcing_ratio = 0.7
latent_dim = 512

# Criar o dataset completo e dividir em treino e validação (80% / 20%)
full_dataset = Seq2SeqDataset(label_train, decoder_input_data, decoder_target_data)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)

# Tamanhos dos vocabulários
label_vocab_size = len(label_tokenizer.word_index) + 1
desc_vocab_size  = len(description_tokenizer.word_index) + 1

# Definição do Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (h, c) = self.lstm(embedded)
        return h, c

# Definição do Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, h, c):
        embedded = self.embedding(x)
        output, (h, c) = self.lstm(embedded, (h, c))
        prediction = self.fc(output)
        return prediction, h, c

# Definição do modelo Seq2Seq com teacher forcing ajustável
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, encoder_inputs, decoder_inputs, teacher_forcing_ratio=0.5):
        batch_size = encoder_inputs.size(0)
        seq_len = decoder_inputs.size(1)
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, seq_len, vocab_size).to(self.device)
        
        h, c = self.encoder(encoder_inputs)
        # Iniciar com o token de início (primeiro token do decoder_inputs)
        input_token = decoder_inputs[:, 0].unsqueeze(1)
        for t in range(seq_len):
            output, h, c = self.decoder(input_token, h, c)
            outputs[:, t, :] = output.squeeze(1)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if t + 1 < seq_len:
                if use_teacher_forcing:
                    input_token = decoder_inputs[:, t+1].unsqueeze(1)
                else:
                    input_token = output.argmax(2)
        return outputs

# Instanciar os módulos
encoder = Encoder(label_vocab_size, latent_dim, latent_dim)
decoder = Decoder(desc_vocab_size, latent_dim, latent_dim)
model = Seq2Seq(encoder, decoder, device).to(device)

# Configurar otimizador e função de perda
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Função de treinamento com contagem de tempo por batch, por epoch e total
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, teacher_forcing_ratio):
    overall_start = time.time()
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for batch_idx, (encoder_inputs, decoder_inputs, decoder_targets) in enumerate(train_loader):
            batch_start = time.time()
            
            encoder_inputs = encoder_inputs.to(device)
            decoder_inputs = decoder_inputs.to(device)
            decoder_targets = decoder_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(encoder_inputs, decoder_inputs, teacher_forcing_ratio)
            loss = criterion(outputs.view(-1, outputs.size(-1)), decoder_targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_end = time.time()
            print(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Batch Time: {batch_end - batch_start:.2f}s")
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validação sem teacher forcing
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for encoder_inputs, decoder_inputs, decoder_targets in val_loader:
                encoder_inputs = encoder_inputs.to(device)
                decoder_inputs = decoder_inputs.to(device)
                decoder_targets = decoder_targets.to(device)
                outputs = model(encoder_inputs, decoder_inputs, teacher_forcing_ratio=0.0)
                loss = criterion(outputs.view(-1, outputs.size(-1)), decoder_targets.view(-1))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        epoch_end = time.time()
        print(f"Epoch {epoch+1}/{num_epochs} finished in {epoch_end - epoch_start:.2f}s - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # Salvar o modelo com melhor loss de validação (pesos separados)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
            }, './models/seq2seq_model_torch.pt')
    overall_end = time.time()
    print(f"Treinamento concluído em {overall_end - overall_start:.2f}s. Best Val Loss: {best_val_loss:.4f}")

# Iniciar o treinamento e contabilizar o tempo total do ficheiro
train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, teacher_forcing_ratio)

# reinamento concluído em 1382.82s. Best Val Loss: 7.0140 | 0.384117 Horas