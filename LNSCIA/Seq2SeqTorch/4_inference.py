
import torch
import pickle
import numpy as np
from train import Encoder, Decoder, Seq2Seq
from tokenizer_utils import SimpleTokenizer

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar tokenizadores
with open('./models/label_tokenizer.pkl', 'rb') as f:
    label_tokenizer = pickle.load(f)
with open('./models/description_tokenizer.pkl', 'rb') as f:
    description_tokenizer = pickle.load(f)

# Parâmetros do modelo
label_vocab_size = len(label_tokenizer.word_index) + 1
desc_vocab_size = len(description_tokenizer.word_index) + 1

# Criar modelo e carregar pesos
encoder = Encoder(label_vocab_size, emb_dim=300, hidden_dim=256).to(device)
decoder = Decoder(desc_vocab_size, emb_dim=300, enc_hidden_dim=256, dec_hidden_dim=512).to(device)
model = Seq2Seq(encoder, decoder).to(device)
model.load_state_dict(torch.load('./models/best_model.pt', map_location=device))
model.eval()

# Função principal de geração
def generate_description(labels):
    # Verificar palavras OOV
    known_words = set(label_tokenizer.word_index.keys())
    oov = [word for word in labels if word.lower() not in known_words]
    if oov:
        print("⚠️ Atenção: As seguintes palavras não existem no vocabulário e serão tratadas como <UNK>:", oov)

    # Pre-processar labels
    label_str = " ".join(labels).lower()
    label_seq = label_tokenizer.texts_to_sequences([label_str])
    label_seq = torch.LongTensor(label_seq).to(device)

    # Inicializar
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(label_seq)
        input_token = torch.LongTensor([[description_tokenizer.word_index['startseq']]]).to(device)

        output_sentence = []
        for _ in range(50):
            output, hidden, cell = model.decoder(input_token.squeeze(1), hidden, cell, encoder_outputs)
            top1 = output.argmax(1).item()
            if top1 == description_tokenizer.word_index.get('endseq'):
                break
            word = description_tokenizer.index_word.get(top1, '')
            output_sentence.append(word)
            input_token = torch.LongTensor([[top1]]).to(device)

    return " ".join(output_sentence)

# Exemplo de uso
if __name__ == "__main__":
    example = ["céu limpo", "sinal de stop", "carro", "peão", "passadeira"]
    print("Labels:", example)
    print("Descrição gerada:", generate_description(example))                                                                                                                                