import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Função nomeada para expandir a dimensão
def expand_dims(x):
    return tf.expand_dims(x, axis=1)

# Carregar tokenizadores e dados
with open('./models/label_tokenizer.pkl', 'rb') as f:
    label_tokenizer = pickle.load(f)
with open('./models/description_tokenizer.pkl', 'rb') as f:
    description_tokenizer = pickle.load(f)

label_train = np.load('./models/label_train.npy')
desc_train = np.load('./models/desc_train.npy')

# Parâmetros
latent_dim = 512
label_vocab_size = len(label_tokenizer.word_index) + 1
desc_vocab_size = len(description_tokenizer.word_index) + 1

# Preparação dos dados para o Decoder
decoder_input_data = desc_train[:, :-1]
decoder_target_data = desc_train[:, 1:]
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# Encoder
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
encoder_embedding_layer = Embedding(label_vocab_size, latent_dim, mask_zero=True)
encoder_embedding = encoder_embedding_layer(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_sequences=True, return_state=True)(encoder_embedding)

# Obter a máscara do encoder e expandi-la usando uma camada Lambda com função nomeada
encoder_mask = encoder_embedding_layer.compute_mask(encoder_inputs)
if encoder_mask is not None:
    encoder_mask = Lambda(expand_dims)(encoder_mask)  # Shape: (batch, 1, steps)

# Decoder
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
decoder_embedding = Embedding(desc_vocab_size, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm_outputs, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(
    decoder_embedding, initial_state=[state_h, state_c]
)

# Atenção com máscara ajustada
attention_layer = Attention()
attention_output = attention_layer(
    [decoder_lstm_outputs, encoder_outputs],
    mask=[None, encoder_mask]
)

# Concatenação da saída do decoder com a saída da atenção
decoder_concat_input = Concatenate(axis=-1)([decoder_lstm_outputs, attention_output])

# Camada de saída
decoder_dense = Dense(desc_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat_input)

# Definição e compilação do modelo
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(
    [label_train, decoder_input_data], decoder_target_data,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3)
    ]
)

# Salvar o modelo treinado
model.save('./models/seq2seq_attention.keras')