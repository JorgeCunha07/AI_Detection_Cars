import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling1D, Lambda, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Carregar os tokenizadores
with open('./models/label_tokenizer.pkl', 'rb') as f:
    label_tokenizer = pickle.load(f)
with open('./models/description_tokenizer.pkl', 'rb') as f:
    description_tokenizer = pickle.load(f)

# Carregar os dados pré-processados
label_train = np.load('./models/label_train.npy')
label_test = np.load('./models/label_test.npy')
desc_train = np.load('./models/desc_train.npy')
desc_test = np.load('./models/desc_test.npy')

# Parâmetros do modelo
latent_dim = 256  # Dimensão interna (número de filtros usados nas camadas convolucionais)
label_vocab_size = len(label_tokenizer.word_index) + 1
desc_vocab_size = len(description_tokenizer.word_index) + 1

# Preparar os dados do decoder: entradas e targets (deslocados)
decoder_input_data = desc_train[:, :-1]
decoder_target_data = desc_train[:, 1:]
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# ======================
# Modelo de Treinamento
# ======================

# Definir entradas do encoder e decoder
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
decoder_inputs = Input(shape=(None,), name='decoder_inputs')

# Camadas de embedding
encoder_embedding = Embedding(input_dim=label_vocab_size,
                              output_dim=latent_dim,
                              mask_zero=True,
                              name='encoder_embedding')(encoder_inputs)
decoder_embedding = Embedding(input_dim=desc_vocab_size,
                              output_dim=latent_dim,
                              mask_zero=True,
                              name='decoder_embedding')(decoder_inputs)

# ----- Encoder CNN -----
# Aplicar duas camadas convolucionais e obter um vetor de contexto fixo via pooling
conv_enc = Conv1D(filters=latent_dim, kernel_size=3, padding='same', activation='relu', name='conv1d_enc_1')(encoder_embedding)
conv_enc = Conv1D(filters=latent_dim, kernel_size=3, padding='same', activation='relu', name='conv1d_enc_2')(conv_enc)
context_vector = GlobalMaxPooling1D(name='global_pooling')(conv_enc)  # Forma: (batch_size, latent_dim)

# ----- Decoder CNN -----
# Função para repetir o vetor de contexto para cada timestep do decoder
def repeat_context(inputs):
    context, decoder_seq = inputs
    repeated = tf.repeat(tf.expand_dims(context, axis=1), tf.shape(decoder_seq)[1], axis=1)
    return repeated

context_repeated = Lambda(repeat_context, name='repeat_context')([context_vector, decoder_inputs])

# Concatena o embedding do decoder com o contexto repetido
decoder_combined = Concatenate(name='concat_context')([decoder_embedding, context_repeated])

# Aplicar camadas convolucionais com padding causal (para preservar a propriedade autoregressiva)
conv_dec = Conv1D(filters=latent_dim, kernel_size=3, padding='causal', activation='relu', name='conv1d_2')(decoder_combined)
conv_dec = Conv1D(filters=latent_dim, kernel_size=3, padding='causal', activation='relu', name='conv1d_3')(conv_dec)

# Camada densa final para prever a palavra em cada timestep
decoder_outputs = Dense(desc_vocab_size, activation='softmax', name='decoder_dense')(conv_dec)

# Modelo Seq2Seq CNN para treinamento
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks para treino
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3)
]

# Treinar o modelo
model.fit(
    [label_train, decoder_input_data],
    decoder_target_data,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks
)

# Salvar o modelo treinado
model.save('./models/seq2seq_model_cnn.keras')

# ============================
# Modelos para Inferência
# ============================

# Modelo do encoder para inferência: recebe a sequência de labels e retorna o vetor de contexto
encoder_model = Model(encoder_inputs, context_vector)

# Modelo do decoder para inferência:
# Entrada: sequência do decoder e o vetor de contexto do encoder
decoder_inputs_inf = Input(shape=(None,), name='decoder_inputs_inf')
decoder_embedding_inf = model.get_layer('decoder_embedding')(decoder_inputs_inf)

decoder_context_input = Input(shape=(latent_dim,), name='decoder_context_input')
context_repeated_inf = Lambda(repeat_context, name='repeat_context_inf')([decoder_context_input, decoder_inputs_inf])

decoder_combined_inf = Concatenate(name='concat_context_inf')([decoder_embedding_inf, context_repeated_inf])

# Utilize as camadas do decoder que foram aplicadas durante o treinamento
conv_dec_inf = model.get_layer('conv1d_2')(decoder_combined_inf)
conv_dec_inf = model.get_layer('conv1d_3')(conv_dec_inf)
decoder_outputs_inf = model.get_layer('decoder_dense')(conv_dec_inf)

# Seleciona o output do último timestep
def last_timestep(x):
    return x[:, -1, :]

last_output = Lambda(last_timestep, name='last_timestep')(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs_inf, decoder_context_input],
    last_output
)

# ============================
# Função de Inferência
# ============================
def decode_sequence(input_seq, max_length=50):
    # Obtém o vetor de contexto a partir do encoder
    context_val = encoder_model.predict(input_seq)
    if 'startseq' not in description_tokenizer.word_index:
        raise ValueError("Token 'startseq' não encontrado no vocabulário.")
    # Inicializa a sequência com o token de início
    target_seq = np.array([[description_tokenizer.word_index['startseq']]])
    decoded_sentence = []
    for _ in range(max_length):
        # Prever o próximo token com base na sequência atual e o contexto
        output_tokens = decoder_model.predict([target_seq, context_val])
        sampled_token_index = np.argmax(output_tokens[0])
        sampled_word = description_tokenizer.index_word.get(sampled_token_index, '')
        if sampled_word == 'endseq' or sampled_word == '':
            break
        decoded_sentence.append(sampled_word)
        # Acrescenta o token gerado à sequência do decoder
        target_seq = np.concatenate([target_seq, np.array([[sampled_token_index]])], axis=1)
    return ' '.join(decoded_sentence)

# ============================
# Exemplos de Inferência
# ============================
generated_descriptions = []
print("\nExemplos de inferência:")
for seq in label_test[:10]:
    input_seq = seq.reshape(1, -1)
    decoded_sentence = decode_sequence(input_seq)
    generated_descriptions.append(decoded_sentence)
    print("Gerado:", decoded_sentence)

# ============================
# Funções de Avaliação: BLEU e ROUGE
# ============================
def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())

def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]['rouge-1']['f']

# Calcular métricas para os primeiros exemplos
bleu_scores = []
rouge_scores = []
for i in range(min(len(desc_test), len(generated_descriptions))):
    reference_tokens = [description_tokenizer.index_word.get(idx, '') for idx in desc_test[i] if idx != 0]
    reference_sentence = ' '.join(reference_tokens)
    candidate_sentence = generated_descriptions[i]
    bleu_scores.append(calculate_bleu(reference_sentence, candidate_sentence))
    rouge_scores.append(calculate_rouge(reference_sentence, candidate_sentence))

print(f"\nBLEU Score Médio: {sum(bleu_scores)/len(bleu_scores):.4f}")
print(f"ROUGE Score Médio: {sum(rouge_scores)/len(rouge_scores):.4f}")
