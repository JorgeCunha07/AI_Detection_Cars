import pickle

import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.sequence import pad_sequences

# Carregar modelo treinado
model = load_model('./models/seq2seq_model.keras')

# Carregar tokenizer e parâmetros
with open('./models/label_tokenizer.pkl', 'rb') as f:
    tokenizer_labels = pickle.load(f)

with open('./models/description_tokenizer.pkl', 'rb') as f:
    tokenizer_desc = pickle.load(f)

with open('./models/preprocess_params.pkl', 'rb') as f:
    preprocess_params = pickle.load(f)
    max_labels_len = preprocess_params['max_label_length']
    max_desc_len = preprocess_params['max_desc_length']

index_word = tokenizer_desc.index_word


# Função para gerar texto a partir das labels fornecidas
def generate_text(labels_input):
    # Converte labels numa string unica
    input_text = ", ".join(labels_input)

    # Converte labels numa sequencia numerica
    labels_seq = tokenizer_labels.texts_to_sequences([input_text])
    labels_seq_padded = pad_sequences(labels_seq, maxlen=max_labels_len, padding='post')

    labels_seq_padded = np.array(labels_seq_padded[:1])

    # Define os tokens especiais
    start_token = tokenizer_desc.word_index['startseq']
    end_token = tokenizer_desc.word_index['endseq']

    decoder_input = np.zeros((1, max_desc_len))
    decoder_input[0, 0] = start_token

    generated_text = []

    # Gerar palavra a palavra
    for i in range(1, max_desc_len):
        # Obtem previsoes do modelo
        preds = model.predict([labels_seq_padded, decoder_input])
        token_index = np.argmax(preds[0, i - 1, :])

        if token_index == end_token or token_index == 0:
            break

        word = index_word.get(token_index, '')
        generated_text.append(word)

        # Atualiza o input do decoder para o proximo passo
        decoder_input[0, i] = token_index

    return ' '.join(generated_text)


labels_test = ["nublado", "semaforo", "peao", "passadeira"]
generated_description = generate_text(labels_test)
print(generated_description)
# a cena vento forte situada proximo de um parque a tarde inclui autocarro a circular rapidamente peao a espera junto ao semaforo e infraestruturas como passadeira e sinais de obrigacao sob condicoes de transito moderado
labels_test = ["autocarro", "camião", "semáforo", "obras na via"]
generated_description = generate_text(labels_test)
print(generated_description)
# a cena vento forte situada proximo de um parque a tarde inclui autocarro camiao a circular rapidamente peao a caminhar pela via e infraestruturas como semaforo e sinais de obrigacao sob condicoes de transito leve