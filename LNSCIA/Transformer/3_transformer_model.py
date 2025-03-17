import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Camada customizada para adicionar Positional Encoding
class AddPositionalEncoding(Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super(AddPositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = self.compute_positional_encoding(max_len, d_model)

    def compute_positional_encoding(self, seq_len, d_model):
        positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        dims = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = positions * angle_rates
        # Para índices pares usa sin, para ímpares cos
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # Shape: (1, seq_len, d_model)
        return pos_encoding

    def call(self, x):
        seq_len = tf.shape(x)[1]
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        return x + pos_encoding

    def get_config(self):
        config = super(AddPositionalEncoding, self).get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model,
        })
        return config

# Bloco do Encoder do Transformer
def transformer_encoder_block(x, d_model, num_heads, dff, dropout_rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = Dense(dff, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2

# Bloco do Decoder do Transformer
def transformer_decoder_block(x, enc_output, d_model, num_heads, dff, dropout_rate=0.1):
    attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn1 = Dropout(dropout_rate)(attn1)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn1)
    attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(out1, enc_output)
    attn2 = Dropout(dropout_rate)(attn2)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + attn2)
    ffn_output = Dense(dff, activation='relu')(out2)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out3 = LayerNormalization(epsilon=1e-6)(out2 + ffn_output)
    return out3

# Construção do Modelo Transformer
def build_transformer_model(label_vocab_size, desc_vocab_size, max_labels_len, max_desc_len,
                            d_model=512, num_heads=8, dff=2048, dropout_rate=0.1):
    # Encoder
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    enc_embedding = Embedding(label_vocab_size, d_model)(encoder_inputs)
    enc_embedding = AddPositionalEncoding(max_labels_len, d_model)(enc_embedding)
    enc_output = transformer_encoder_block(enc_embedding, d_model, num_heads, dff, dropout_rate)
    
    # Decoder
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    dec_embedding = Embedding(desc_vocab_size, d_model)(decoder_inputs)
    dec_embedding = AddPositionalEncoding(max_desc_len, d_model)(dec_embedding)
    dec_output = transformer_decoder_block(dec_embedding, enc_output, d_model, num_heads, dff, dropout_rate)
    
    # Camada final para projeção para o vocabulário
    outputs = Dense(desc_vocab_size, activation="softmax")(dec_output)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    return model

# Carregar tokenizadores e parâmetros de pré-processamento
with open('./models/label_tokenizer.pkl', 'rb') as f:
    label_tokenizer = pickle.load(f)
with open('./models/description_tokenizer.pkl', 'rb') as f:
    description_tokenizer = pickle.load(f)
with open('./models/preprocess_params.pkl', 'rb') as f:
    preprocess_params = pickle.load(f)
    max_label_length = preprocess_params['max_label_length']
    max_desc_length = preprocess_params['max_desc_length']

# Carregar dados de treinamento
label_train = np.load('./models/label_train.npy')
desc_train = np.load('./models/desc_train.npy')

# Definir os tamanhos dos vocabulários
label_vocab_size = len(label_tokenizer.word_index) + 1
desc_vocab_size = len(description_tokenizer.word_index) + 1

# Preparar os dados para o decoder: deslocamento da sequência
decoder_input_data = desc_train[:, :-1]
decoder_target_data = desc_train[:, 1:]
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# Construir e compilar o modelo Transformer
transformer_model = build_transformer_model(label_vocab_size, desc_vocab_size, max_label_length, max_desc_length)
transformer_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

transformer_model.summary()

transformer_model.fit(
    [label_train, decoder_input_data], decoder_target_data,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', patience=3)
    ]
)

transformer_model.save('./models/transformer_model.keras')
