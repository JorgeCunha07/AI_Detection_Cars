import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_seq2seq_model
from data_preparation import (
    encoder_input_data,
    decoder_input_data,
    max_decoder_seq_length,
    encoder_vocab_size,
    decoder_vocab_size
)

# Opcional: Ignorar warnings relacionados à máscara
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not support masking.*")

# ----------------------------------------------------
# Preparação dos Dados Alvo para o Treinamento
# ----------------------------------------------------
# O modelo foi construído para gerar (T-1) time steps,
# portanto o target é a sequência do decoder_input_data deslocada para a esquerda.
decoder_target_data = decoder_input_data[:, 1:]  # Shape: (batch, T-1)
decoder_target_data = np.expand_dims(decoder_target_data, -1)  # Agora: (batch, T-1, 1)

# ----------------------------------------------------
# Parâmetros de Treinamento
# ----------------------------------------------------
batch_size = 2
epochs = 50

# ----------------------------------------------------
# Construção e Compilação do Modelo
# ----------------------------------------------------
model = build_seq2seq_model(
    encoder_vocab_size=encoder_vocab_size,
    decoder_vocab_size=decoder_vocab_size,
    max_decoder_seq_length=max_decoder_seq_length,
    embedding_dim=64,
    latent_dim=128
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Exibe o resumo do modelo
model.summary()

# ----------------------------------------------------
# Callbacks para Monitoramento
# ----------------------------------------------------
early_stop = EarlyStopping(monitor='loss', patience=5)
checkpoint_path = "outputs/model_weights/best_model.weights.h5"
os.makedirs("outputs/model_weights", exist_ok=True)

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='loss',
    save_best_only=True,
    save_weights_only=True
)

# ----------------------------------------------------
# Treinamento do Modelo
# ----------------------------------------------------
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stop, checkpoint]
)

# Salva os pesos do último modelo (como backup)
model.save_weights("outputs/model_weights/last_model.weights.h5")
print("Treinamento concluído. Pesos salvos em outputs/model_weights/last_model.weights.h5")
