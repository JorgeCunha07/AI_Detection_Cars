import pickle
import numpy as np
import tensorflow as tf
from keras.api.models import load_model, Model
from keras.api.preprocessing.sequence import pad_sequences

# Define a função customizada repeat_context
def repeat_context(inputs):
    context, decoder_seq = inputs
    repeated = tf.repeat(tf.expand_dims(context, axis=1), tf.shape(decoder_seq)[1], axis=1)
    return repeated

# Carregar o modelo treinado, informando a função customizada
custom_objs = {'repeat_context': repeat_context}
model = load_model('./models/seq2seq_model_cnn.keras', custom_objects=custom_objs, compile=False)

# Carregar tokenizers e parâmetros de pré-processamento
with open('./models/label_tokenizer.pkl', 'rb') as f:
    tokenizer_labels = pickle.load(f)

with open('./models/description_tokenizer.pkl', 'rb') as f:
    tokenizer_desc = pickle.load(f)

with open('./models/preprocess_params.pkl', 'rb') as f:
    preprocess_params = pickle.load(f)
    max_labels_len = preprocess_params['max_label_length']
    max_desc_len = preprocess_params['max_desc_length']

index_word = tokenizer_desc.index_word
latent_dim = 256  # Deve ser igual à dimensão utilizada no treinamento

# --------------------------------------------------
# Reconstrução dos modelos de inferência
# --------------------------------------------------

# Modelo do encoder: recebe a sequência de labels e retorna o vetor de contexto
encoder_inputs = model.input[0]  # Entrada do encoder
context_vector = model.get_layer('global_pooling').output  # Vetor de contexto do encoder
encoder_model = Model(encoder_inputs, context_vector)

# Modelo do decoder para inferência:
decoder_inputs_inf = tf.keras.layers.Input(shape=(None,), name='decoder_inputs_inf')
decoder_embedding_inf = model.get_layer('decoder_embedding')(decoder_inputs_inf)

decoder_context_input = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_context_input')
context_repeated_inf = tf.keras.layers.Lambda(repeat_context, name='repeat_context_inf')(
    [decoder_context_input, decoder_inputs_inf]
)

decoder_combined_inf = tf.keras.layers.Concatenate(name='concat_context_inf')(
    [decoder_embedding_inf, context_repeated_inf]
)

# Utiliza as camadas do decoder (conforme treinamento)
conv_dec_inf = model.get_layer('conv1d_2')(decoder_combined_inf)
conv_dec_inf = model.get_layer('conv1d_3')(conv_dec_inf)
decoder_outputs_inf = model.get_layer('decoder_dense')(conv_dec_inf)

# Seleciona o output do último timestep
def last_timestep(x):
    return x[:, -1, :]

last_output = tf.keras.layers.Lambda(last_timestep, name='last_timestep')(decoder_outputs_inf)

decoder_model = Model([decoder_inputs_inf, decoder_context_input], last_output)

# Função de inferência: gera a descrição a partir da sequência numérica dos labels
def decode_sequence(input_seq, max_length=50):
    # Obtém o vetor de contexto a partir do encoder
    context_val = encoder_model.predict(input_seq)
    if 'startseq' not in tokenizer_desc.word_index:
        raise ValueError("Token 'startseq' não encontrado no vocabulário.")
    # Inicializa a sequência do decoder com o token de início
    target_seq = np.array([[tokenizer_desc.word_index['startseq']]])
    decoded_sentence = []
    for _ in range(max_length):
        output_tokens = decoder_model.predict([target_seq, context_val])
        sampled_token_index = np.argmax(output_tokens[0])
        sampled_word = tokenizer_desc.index_word.get(sampled_token_index, '')
        if sampled_word == 'endseq' or sampled_word == '':
            break
        decoded_sentence.append(sampled_word)
        # Atualiza a sequência do decoder com o token gerado
        target_seq = np.concatenate([target_seq, np.array([[sampled_token_index]])], axis=1)
    return ' '.join(decoded_sentence)

# Função para gerar texto a partir dos labels fornecidos
def generate_text(labels_input):
    # Converte a lista de labels em uma única string
    input_text = ", ".join(labels_input)
    # Converte a string de labels para uma sequência numérica
    labels_seq = tokenizer_labels.texts_to_sequences([input_text])
    labels_seq_padded = pad_sequences(labels_seq, maxlen=max_labels_len, padding='post')
    labels_seq_padded = np.array(labels_seq_padded[:1])
    # Gera a descrição chamando a função de inferência
    generated_text = decode_sequence(labels_seq_padded, max_length=max_desc_len)
    return generated_text

# Exemplo de inferência
labels_test = ["sol", "semaforo", "peao"]
generated_description = generate_text(labels_test)
print(generated_description)

"""
Exemplo de saída:
no contexto em uma rua movimentada, destacam-se camião e semáforo, vistos parados no semáforo, numa altura do dia nublado à noite e com trânsito moderado.
"""
