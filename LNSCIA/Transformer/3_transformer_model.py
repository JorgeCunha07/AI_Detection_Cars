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


'''
PS C:\Users\CarlosMoutinho(11408\Documents\GitHub\AI_Detection_Cars\LNSCIA\Transformer> & "C:/Users/CarlosMoutinho(11408/AppData/Local/Programs/Python/Python311/python.exe" "c:/Users/CarlosMoutinho(11408/Documents/GitHub/AI_Detection_Cars/LNSCIA/Transformer/3_transformer_model.py"
2025-03-18 14:18:58.910099: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 14:18:59.824988: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-18 14:19:02.460541: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:From C:\Users\CarlosMoutinho(11408\AppData\Local\Programs\Python\Python311\Lib\site-packages\keras\src\backend\tensorflow\core.py:219: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ encoder_inputs (InputLayer)   │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, None, 512)         │           9,728 │ encoder_inputs[0][0]       │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_positional_encoding       │ (None, None, 512)         │               0 │ embedding[0][0]            │
│ (AddPositionalEncoding)       │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention          │ (None, None, 512)         │       8,401,408 │ add_positional_encoding[0… │
│ (MultiHeadAttention)          │                           │                 │ add_positional_encoding[0… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, None, 512)         │               0 │ multi_head_attention[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ decoder_inputs (InputLayer)   │ (None, None)              │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, None, 512)         │               0 │ add_positional_encoding[0… │
│                               │                           │                 │ dropout_1[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding_1 (Embedding)       │ (None, None, 512)         │          76,288 │ decoder_inputs[0][0]       │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization           │ (None, None, 512)         │           1,024 │ add[0][0]                  │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_positional_encoding_1     │ (None, None, 512)         │               0 │ embedding_1[0][0]          │
│ (AddPositionalEncoding)       │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, None, 2048)        │       1,050,624 │ layer_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention_1        │ (None, None, 512)         │       8,401,408 │ add_positional_encoding_1… │
│ (MultiHeadAttention)          │                           │                 │ add_positional_encoding_1… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_1 (Dense)               │ (None, None, 512)         │       1,049,088 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_4 (Dropout)           │ (None, None, 512)         │               0 │ multi_head_attention_1[0]… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, None, 512)         │               0 │ dense_1[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_2 (Add)                   │ (None, None, 512)         │               0 │ add_positional_encoding_1… │
│                               │                           │                 │ dropout_4[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_1 (Add)                   │ (None, None, 512)         │               0 │ layer_normalization[0][0], │
│                               │                           │                 │ dropout_2[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_2         │ (None, None, 512)         │           1,024 │ add_2[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_1         │ (None, None, 512)         │           1,024 │ add_1[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention_2        │ (None, None, 512)         │       8,401,408 │ layer_normalization_2[0][… │
│ (MultiHeadAttention)          │                           │                 │ layer_normalization_1[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_6 (Dropout)           │ (None, None, 512)         │               0 │ multi_head_attention_2[0]… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_3 (Add)                   │ (None, None, 512)         │               0 │ layer_normalization_2[0][… │
│                               │                           │                 │ dropout_6[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_3         │ (None, None, 512)         │           1,024 │ add_3[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_2 (Dense)               │ (None, None, 2048)        │       1,050,624 │ layer_normalization_3[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_3 (Dense)               │ (None, None, 512)         │       1,049,088 │ dense_2[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_7 (Dropout)           │ (None, None, 512)         │               0 │ dense_3[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_4 (Add)                   │ (None, None, 512)         │               0 │ layer_normalization_3[0][… │
│                               │                           │                 │ dropout_7[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_4         │ (None, None, 512)         │           1,024 │ add_4[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_4 (Dense)               │ (None, None, 149)         │          76,437 │ layer_normalization_4[0][… │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 29,571,221 (112.81 MB)
 Trainable params: 29,571,221 (112.81 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 268s 21s/step - accuracy: 0.1120 - loss: 8.3697 - val_accuracy: 0.0324 - val_loss: 4.4580 - learning_rate: 0.0010
Epoch 2/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 91s 9s/step - accuracy: 0.1799 - loss: 4.4179 - val_accuracy: 0.2269 - val_loss: 4.2329 - learning_rate: 0.0010
Epoch 3/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 125s 13s/step - accuracy: 0.2299 - loss: 4.2579 - val_accuracy: 0.2269 - val_loss: 4.1933 - learning_rate: 0.0010
Epoch 4/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 99s 11s/step - accuracy: 0.2322 - loss: 4.2235 - val_accuracy: 0.2269 - val_loss: 4.1981 - learning_rate: 0.0010
Epoch 5/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 61s 4s/step - accuracy: 0.2308 - loss: 4.2126 - val_accuracy: 0.2269 - val_loss: 4.1715 - learning_rate: 0.0010
Epoch 6/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 86s 9s/step - accuracy: 0.2296 - loss: 4.2107 - val_accuracy: 0.2269 - val_loss: 4.1715 - learning_rate: 0.0010
Epoch 7/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 98s 10s/step - accuracy: 0.2325 - loss: 4.2005 - val_accuracy: 0.2269 - val_loss: 4.1718 - learning_rate: 0.0010
Epoch 8/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 33s 3s/step - accuracy: 0.2292 - loss: 4.2070 - val_accuracy: 0.2269 - val_loss: 4.1742 - learning_rate: 0.0010
Epoch 9/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 101s 11s/step - accuracy: 0.2306 - loss: 4.2017 - val_accuracy: 0.2269 - val_loss: 4.1701 - learning_rate: 1.0000e-04
Epoch 10/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 31s 3s/step - accuracy: 0.2293 - loss: 4.1966 - val_accuracy: 0.2269 - val_loss: 4.1686 - learning_rate: 1.0000e-04
Epoch 11/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 52s 5s/step - accuracy: 0.2319 - loss: 4.1979 - val_accuracy: 0.2269 - val_loss: 4.1687 - learning_rate: 1.0000e-04
Epoch 12/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 30s 3s/step - accuracy: 0.2316 - loss: 4.1948 - val_accuracy: 0.2269 - val_loss: 4.1686 - learning_rate: 1.0000e-04
Epoch 13/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 71s 8s/step - accuracy: 0.2297 - loss: 4.1993 - val_accuracy: 0.2269 - val_loss: 4.1688 - learning_rate: 1.0000e-04
Epoch 14/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 32s 3s/step - accuracy: 0.2278 - loss: 4.2073 - val_accuracy: 0.2269 - val_loss: 4.1688 - learning_rate: 1.0000e-05
Epoch 15/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 30s 3s/step - accuracy: 0.2294 - loss: 4.1963 - val_accuracy: 0.2269 - val_loss: 4.1689 - learning_rate: 1.0000e-05
Epoch 16/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 39s 4s/step - accuracy: 0.2298 - loss: 4.1995 - val_accuracy: 0.2269 - val_loss: 4.1690 - learning_rate: 1.0000e-05
Epoch 17/50
10/10 ━━━━━━━━━━━━━━━━━━━━ 65s 3s/step - accuracy: 0.2345 - loss: 4.1832 - val_accuracy: 0.2269 - val_loss: 4.1690 - learning_rate: 1.0000e-06

'''