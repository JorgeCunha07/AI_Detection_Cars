import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Registra a função customizada para deserialização
@tf.keras.utils.register_keras_serializable()
def expand_dims(x):
    return tf.expand_dims(x, axis=1)

# Carregar o modelo e informar a função customizada em custom_objects
model = load_model('./models/seq2seq_attention.keras', custom_objects={'expand_dims': expand_dims})

# Carregar tokenizadores e parâmetros
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
    # Converte labels numa única string
    input_text = ", ".join(labels_input)

    # Converte labels numa sequência numérica
    labels_seq = tokenizer_labels.texts_to_sequences([input_text])
    labels_seq_padded = pad_sequences(labels_seq, maxlen=max_labels_len, padding='post')
    labels_seq_padded = np.array(labels_seq_padded[:1])

    # Define os tokens especiais
    start_token = tokenizer_desc.word_index['startseq']
    end_token = tokenizer_desc.word_index['endseq']

    decoder_input = np.zeros((1, max_desc_len))
    decoder_input[0, 0] = start_token

    generated_text = []

    # Geração palavra a palavra
    for i in range(1, max_desc_len):
        preds = model.predict([labels_seq_padded, decoder_input])
        token_index = np.argmax(preds[0, i - 1, :])
        if token_index == end_token or token_index == 0:
            break
        word = index_word.get(token_index, '')
        generated_text.append(word)
        decoder_input[0, i] = token_index

    return ' '.join(generated_text)

labels_test = ["carro", "peão", "semáforo", "sinal de stop"]
generated_description = generate_text(labels_test)
print(generated_description)

labels_test = ["autocarro", "passadeira", "limite de velocidade", "engarrafamento"]
generated_description = generate_text(labels_test)
print(generated_description)

labels_test = ["camião", "ciclista", "sinal de proibido", "vento forte"]
generated_description = generate_text(labels_test)
print(generated_description)

labels_test = ["carro", "peão", "passadeira", "trânsito intenso"]
generated_description = generate_text(labels_test)
print(generated_description)

labels_test = ["autocarro", "camião", "semáforo", "obras na via"]
generated_description = generate_text(labels_test)
print(generated_description)