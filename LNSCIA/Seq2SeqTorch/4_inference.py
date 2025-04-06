import torch
import pickle
import numpy as np
from tokenizer_utils import pad_sequences, SimpleTokenizer
from train import Encoder, Decoder, Seq2Seq

# ============================
# Parâmetros e dispositivo
# ============================
MODEL_DIR = './models/'
EMB_DIM = 300
HIDDEN_DIM = 512
DROPOUT = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# ============================
# Carregar tokenizadores e modelo
# ============================
with open(f"{MODEL_DIR}/label_tokenizer.pkl", 'rb') as f:
    label_tokenizer = pickle.load(f)

with open(f"{MODEL_DIR}/description_tokenizer.pkl", 'rb') as f:
    description_tokenizer = pickle.load(f)

with open(f"{MODEL_DIR}/preprocess_params.pkl", 'rb') as f:
    params = pickle.load(f)

max_label_length = params['max_label_length']
max_desc_length = params['max_desc_length']

label_vocab_size = len(label_tokenizer.word_index) + 1
desc_vocab_size = max(description_tokenizer.word_index.values()) + 1

# Instanciar modelo
enc = Encoder(label_vocab_size, EMB_DIM, HIDDEN_DIM, dropout=DROPOUT).to(device)
dec = Decoder(desc_vocab_size, EMB_DIM, HIDDEN_DIM, HIDDEN_DIM, dropout=DROPOUT).to(device)
model = Seq2Seq(enc, dec).to(device)
model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pt", map_location=device))
model.eval()

# ============================
# Função de geração de descrição
# ============================
def generate_description(labels_input):
    labels_clean = ' '.join(labels_input).lower()
    seq = label_tokenizer.texts_to_sequences([labels_clean])
    padded = pad_sequences(seq, maxlen=max_label_length, padding='post')
    src = torch.LongTensor(padded).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

        start_token = description_tokenizer.word_index['startseq']
        end_token = description_tokenizer.word_index['endseq']

        input_token = torch.LongTensor([start_token]).to(device)
        generated = []

        for _ in range(max_desc_length):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1).item()
            if top1 == end_token or top1 == 0:
                break
            word = description_tokenizer.index_word.get(top1, '')
            generated.append(word)
            input_token = torch.LongTensor([top1]).to(device)

    return ' '.join(generated)

# ============================
# Exemplo de uso
# ============================
if __name__ == '__main__':
    example = ["carro", "passadeira", "semáforo", "noite"]
    print("Labels:", example)
    description = generate_description(example)
    print("Descrição gerada:", description)