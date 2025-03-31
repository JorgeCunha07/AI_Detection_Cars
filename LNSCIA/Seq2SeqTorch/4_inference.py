import os
import pickle
import numpy as np
import torch
import torch.nn as nn

# Carregar tokenizadores e parâmetros de pré-processamento
with open('./models/label_tokenizer.pkl', 'rb') as f:
    tokenizer_labels = pickle.load(f)
with open('./models/description_tokenizer.pkl', 'rb') as f:
    tokenizer_desc = pickle.load(f)
with open('./models/preprocess_params.pkl', 'rb') as f:
    preprocess_params = pickle.load(f)
max_labels_len = preprocess_params['max_label_length']
max_desc_len = preprocess_params['max_desc_length']

index_word = tokenizer_desc.index_word

latent_dim = 512
label_vocab_size = len(tokenizer_labels.word_index) + 1
desc_vocab_size  = len(tokenizer_desc.word_index) + 1

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

# Não precisamos de uma classe Seq2Seq completa para inferência
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(label_vocab_size, latent_dim, latent_dim).to(device)
decoder = Decoder(desc_vocab_size, latent_dim, latent_dim).to(device)

# Carregar os pesos salvos separadamente
checkpoint = torch.load('./models/seq2seq_model_torch.pt', map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
encoder.eval()
decoder.eval()

# Função para gerar texto a partir das labels fornecidas
def generate_text(labels_input, max_length=50):
    # Converter a lista de labels numa string
    input_text = ", ".join(labels_input)
    # Converter para sequência numérica usando o tokenizer dos labels
    labels_seq = tokenizer_labels.texts_to_sequences([input_text])
    # Usar a função de padding do Keras (ou sua própria implementação)
    from keras.api.preprocessing.sequence import pad_sequences
    labels_seq_padded = pad_sequences(labels_seq, maxlen=max_labels_len, padding='post')
    labels_seq_padded = labels_seq_padded[0]
    
    encoder_input = torch.LongTensor(labels_seq_padded).unsqueeze(0).to(device)
    with torch.no_grad():
        h, c = encoder(encoder_input)
    start_token = tokenizer_desc.word_index.get('startseq')
    end_token = tokenizer_desc.word_index.get('endseq')
    input_token = torch.LongTensor([[start_token]]).to(device)
    generated_words = []
    with torch.no_grad():
        for _ in range(max_length):
            output, h, c = decoder(input_token, h, c)
            token_id = output.squeeze(1).argmax(1).item()
            if token_id == end_token or token_id == 0:
                break
            word = index_word.get(token_id, '')
            generated_words.append(word)
            input_token = torch.LongTensor([[token_id]]).to(device)
    return ' '.join(generated_words)

# Exemplos de inferência
labels_test = ["nublado", "semáforo", "peão", "passadeira"]
generated_description = generate_text(labels_test)
print("Gerado:", generated_description)

# Gerado: a cena de proximo de um parque a a meio da manha sob um ceu nublado ciclista peao podem ser vistas a aguardar para atravessar enquanto carro passam a circular lentamente com passagem de nivel e sinal de stop em destaque numa situacao de transito moderado

labels_test = ["autocarro", "camião", "semáforo", "sinal de stop"]
generated_description = generate_text(labels_test)
print("Gerado:", generated_description)

# Gerado: a cena de proximo de uma estacao de metro a tarde inclui camiao a circular rapidamente ciclista a caminhar pela via e infraestruturas como semaforo e sinal de proibido sob condicoes de transito intenso

labels_test = ["carro", "peão", "passadeira", "sinal de limite de velocidade"]
generated_description = generate_text(labels_test)
print("Gerado:", generated_description)

# Gerado: a cena de proximo de uma estacao de metro a tarde inclui camiao a circular rapidamente ciclista a caminhar pela via e infraestruturas como semaforo e sinal de proibido sob condicoes de transito intenso

labels_test = ["autocarro", "ciclista", "semáforo", "obras na via"]
generated_description = generate_text(labels_test)
print("Gerado:", generated_description)

# Gerado: a cena de proximo de uma estacao de metro a tarde inclui camiao a circular rapidamente ciclista a caminhar pela via e infraestruturas como semaforo e sinal de proibido sob condicoes de transito intenso

labels_test = ["camião", "peão", "sinal de limite de velocidade"]
generated_description = generate_text(labels_test)
print("Gerado:", generated_description)

# Gerado: a cena de proximo de um parque a a tarde sob um ceu nublado ciclista podem ser vistas a espera junto ao semaforo enquanto carro passam a circular lentamente com passagem de nivel e sinal de stop em destaque numa situacao de transito moderado

labels_test = ["carro", "ciclista", "passadeira", "sinal de passadeira"]
generated_description = generate_text(labels_test)
print("Gerado:", generated_description)

# Gerado: a cena de proximo de um parque a a tarde sob um ceu nublado ciclista podem ser vistas a aguardar para atravessar enquanto carro passam a circular lentamente com passagem de nivel e sinal de stop em destaque numa situacao de transito moderado