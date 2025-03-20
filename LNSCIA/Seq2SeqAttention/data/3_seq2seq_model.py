import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
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
latent_dim = 512  # Dimensão do embedding definida para 512
label_vocab_size = len(label_tokenizer.word_index) + 1
desc_vocab_size = len(description_tokenizer.word_index) + 1

# Preparar os dados do decoder: entradas e targets (deslocados)
decoder_input_data = desc_train[:, :-1]
decoder_target_data = desc_train[:, 1:]
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# Definir entradas do encoder e decoder
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
decoder_inputs = Input(shape=(None,), name='decoder_inputs')

# Camadas de embedding
encoder_embedding_layer = Embedding(input_dim=label_vocab_size,
                                    output_dim=latent_dim,
                                    mask_zero=True,
                                    name='encoder_embedding')
decoder_embedding_layer = Embedding(input_dim=desc_vocab_size,
                                    output_dim=latent_dim,
                                    mask_zero=True,
                                    name='decoder_embedding')

# Encoder
encoder_embedding = encoder_embedding_layer(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(desc_vocab_size, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# Modelo Seq2Seq para treinamento
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
model.save('./models/seq2seq_model.keras')

# Construir o modelo do encoder para inferência
encoder_model = Model(encoder_inputs, encoder_states)

# Construir o modelo do decoder para inferência
decoder_state_input_h = Input(shape=(latent_dim,), name='input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_inf = decoder_embedding_layer(decoder_inputs)
decoder_outputs_inf, h_inf, c_inf = decoder_lstm(decoder_embedding_inf, initial_state=decoder_states_inputs)
decoder_states_inf = [h_inf, c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

# Função de inferência
def decode_sequence(input_seq, max_length=50):
    states_value = encoder_model.predict(input_seq)
    if 'startseq' not in description_tokenizer.word_index:
        raise ValueError("Token 'startseq' não encontrado no vocabulário.")
    target_seq = np.array([[description_tokenizer.word_index['startseq']]])
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = description_tokenizer.index_word.get(sampled_token_index, '')
        if sampled_word == 'endseq' or len(decoded_sentence) >= max_length:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]
    return ' '.join(decoded_sentence)

# Exemplos de inferência
generated_descriptions = []
print("\nExemplos de inferência:")
for seq in label_test[:10]:
    input_seq = seq.reshape(1, -1)
    decoded_sentence = decode_sequence(input_seq)
    generated_descriptions.append(decoded_sentence)
    print("Gerado:", decoded_sentence)

# Funções de avaliação: BLEU e ROUGE
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
