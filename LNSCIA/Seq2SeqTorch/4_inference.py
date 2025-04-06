import torch
import pickle
import numpy as np
import os
from tokenizer_utils import pad_sequences
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

# ============================
# Carregar embeddings FastText
# ============================
embedding_weights = np.load(os.path.join(MODEL_DIR, 'fasttext_embeddings.npy'))
desc_vocab_size = embedding_weights.shape[0]
embedding_layer = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weights), freeze=False, padding_idx=0)

# Instanciar modelo com embeddings carregados
enc = Encoder(label_vocab_size, EMB_DIM, HIDDEN_DIM, dropout=DROPOUT).to(device)
dec = Decoder(desc_vocab_size, EMB_DIM, HIDDEN_DIM, HIDDEN_DIM, dropout=DROPOUT).to(device)
dec.embedding = embedding_layer
model = Seq2Seq(enc, dec).to(device)
model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pt", map_location=device))
model.eval()


# ============================
# Geração com beam search
# ============================
def generate_description_beam(labels_input, beam_width=3):
    labels_clean = ' '.join(labels_input).lower()
    seq = label_tokenizer.texts_to_sequences([labels_clean])
    padded = pad_sequences(seq, maxlen=max_label_length, padding='post')
    src = torch.LongTensor(padded).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

        start_token = description_tokenizer.word_index['startseq']
        end_token = description_tokenizer.word_index['endseq']

        sequences = [[[], 0.0, torch.LongTensor([start_token]).to(device), hidden, cell]]

        for _ in range(max_desc_length):
            all_candidates = []
            for seq_tokens, score, input_token, h, c in sequences:
                output, h_new, c_new = model.decoder(input_token, h, c, encoder_outputs)
                log_probs = torch.log_softmax(output, dim=1)
                topk_probs, topk_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    idx = topk_indices[0][i].item()
                    prob = topk_probs[0][i].item()
                    new_seq = seq_tokens + [idx]
                    new_score = (score + prob) / len(new_seq)
                    all_candidates.append([new_seq, new_score, torch.LongTensor([idx]).to(device), h_new, c_new])

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            if all(seq and seq[-1] == end_token for seq, *_ in sequences):
                break

        best_seq = sequences[0][0]
        words = [description_tokenizer.index_word.get(idx, '') for idx in best_seq if idx != end_token]
        return ' '.join(words)


# ... (mantém-se o restante código acima)

# ============================
# Versão sem FastText (baseline)
# ============================
def generate_description_baseline(labels_input, beam_width=3):
    # Usar um decoder com embeddings aleatórios para comparação
    temp_dec = Decoder(desc_vocab_size, EMB_DIM, HIDDEN_DIM, HIDDEN_DIM, dropout=DROPOUT).to(device)
    baseline_model = Seq2Seq(enc, temp_dec).to(device)
    baseline_model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pt", map_location=device))
    baseline_model.eval()

    labels_clean = ' '.join(labels_input).lower()
    seq = label_tokenizer.texts_to_sequences([labels_clean])
    padded = pad_sequences(seq, maxlen=max_label_length, padding='post')
    src = torch.LongTensor(padded).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = baseline_model.encoder(src)

        start_token = description_tokenizer.word_index['startseq']
        end_token = description_tokenizer.word_index['endseq']

        sequences = [[[], 0.0, torch.LongTensor([start_token]).to(device), hidden, cell]]

        for _ in range(max_desc_length):
            all_candidates = []
            for seq_tokens, score, input_token, h, c in sequences:
                output, h_new, c_new = baseline_model.decoder(input_token, h, c, encoder_outputs)
                log_probs = torch.log_softmax(output, dim=1)
                topk_probs, topk_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    idx = topk_indices[0][i].item()
                    prob = topk_probs[0][i].item()
                    new_seq = seq_tokens + [idx]
                    new_score = (score + prob) / len(new_seq)
                    all_candidates.append([new_seq, new_score, torch.LongTensor([idx]).to(device), h_new, c_new])

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            if all(seq and seq[-1] == end_token for seq, *_ in sequences):
                break

        best_seq = sequences[0][0]
        words = [description_tokenizer.index_word.get(idx, '') for idx in best_seq if idx != end_token]
        return ' '.join(words)


# ============================
# Comparação FastText vs Baseline
# ============================
if __name__ == '__main__':
    example = ["carro", "passadeira", "semaforo", "noite"]
    print("Labels:", example)

    fasttext_desc = generate_description_beam(example, beam_width=5)
    print("[FastText]  Descrição:", fasttext_desc)

    baseline_desc = generate_description_baseline(example, beam_width=5)
    print("[Baseline]  Descrição:", baseline_desc)

# Labels: ['carro', 'passadeira', 'semaforo', 'noite']
# Descrição gerada: de está parado no semáforo da passadeira no semáforo da noite. noite.
