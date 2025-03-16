import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Define o caminho do arquivo de dados
data_file = os.path.join("data", "processed", "synthetic_dataset.tsv")

# Lê o dataset a partir do arquivo TSV
df = pd.read_csv(data_file, delimiter="\t")

# Certifique-se de que as colunas estão com os nomes corretos:
# Espera-se que o arquivo possua as colunas: "input_keywords" e "output_description"
print("Colunas do dataset:", df.columns.tolist())

# Extrai os textos de entrada e saída
input_texts = df["input_keywords"].astype(str).tolist()
output_texts = df["output_description"].astype(str).tolist()

# Adiciona tokens especiais à saída: <start> no início e <end> no fim
output_texts = ["<start> " + text + " <end>" for text in output_texts]

# --- Tokenização dos Dados de Entrada ---
# Use filters='' para não remover nenhum caractere (opcional, dependendo da sua necessidade)
input_tokenizer = Tokenizer(filters='')
input_tokenizer.fit_on_texts(input_texts)
encoder_vocab_size = len(input_tokenizer.word_index) + 1  # +1 para o 0 (padding)

# Converte os textos de entrada em sequências numéricas
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
max_encoder_seq_length = max(len(seq) for seq in input_sequences)
encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')

# --- Tokenização dos Dados de Saída ---
target_tokenizer = Tokenizer(filters='')
target_tokenizer.fit_on_texts(output_texts)
decoder_vocab_size = len(target_tokenizer.word_index) + 1

# Converte os textos de saída em sequências numéricas
output_sequences = target_tokenizer.texts_to_sequences(output_texts)
max_decoder_seq_length = max(len(seq) for seq in output_sequences)
decoder_input_data = pad_sequences(output_sequences, maxlen=max_decoder_seq_length, padding='post')

# (Opcional) Salva os tokenizers e arrays processados para uso futuro
os.makedirs("outputs", exist_ok=True)
with open("outputs/input_tokenizer.pkl", "wb") as f:
    pickle.dump(input_tokenizer, f)
with open("outputs/target_tokenizer.pkl", "wb") as f:
    pickle.dump(target_tokenizer, f)
np.save("outputs/encoder_input_data.npy", encoder_input_data)
np.save("outputs/decoder_input_data.npy", decoder_input_data)

# Exibe informações importantes
print("\nTamanho do vocabulário de entrada:", encoder_vocab_size)
print("Tamanho do vocabulário de saída:", decoder_vocab_size)
print("Máximo comprimento das sequências de entrada:", max_encoder_seq_length)
print("Máximo comprimento das sequências de saída:", max_decoder_seq_length)

print("\nExemplo de sequência de entrada (números):")
print(encoder_input_data[:2])
print("\nExemplo de sequência de saída (números):")
print(decoder_input_data[:2])
