import os
import json
import re
import unicodedata
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Verifica se há GPU disponível e configura o crescimento de memória
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU(s) detectada(s) e configurada(s).")
    except RuntimeError as e:
        print("Erro ao configurar GPU: ", e)
else:
    print("Nenhuma GPU detectada, usando CPU.")

# Função para limpar a pasta "./models/"
def clear_models_folder(folder_path='./models/'):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    # Se houver subpastas, removê-las recursivamente
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Não foi possível remover {file_path}. Motivo: {e}')
    else:
        os.makedirs(folder_path)

# Limpar a pasta "./models/" antes de iniciar
clear_models_folder('./models/')

# Função para limpar e normalizar o texto:
def clean_text(text):
    # Remove acentos (normalização Unicode)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8', 'ignore')
    # Converte para minúsculas
    text = text.lower()
    # Remove caracteres que não sejam letras, números ou espaços
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Substitui múltiplos espaços por um único espaço e remove espaços nas extremidades
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Carregar o dataset (arquivo JSON com dados sintéticos e reais)
with open('./data/synthetic_and_real_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

labels_text = []      # Para armazenar os rótulos (labels) em forma de string
descriptions = []     # Para armazenar as descrições

for d in data:
    # Processar a descrição:
    # Se existir o campo "description", usá-lo; caso contrário, usa "trecho"
    desc = d.get("description", d.get("trecho", ""))
    # Remover os tokens especiais temporariamente para limpeza
    desc = desc.replace("startseq", "").replace("endseq", "").strip()
    # Limpar e normalizar o texto
    desc_clean = clean_text(desc)
    # Re-adicionar os tokens especiais de início e fim
    desc_clean = "startseq " + desc_clean + " endseq"
    descriptions.append(desc_clean)
    
    # Processar os rótulos:
    # Se existir o campo "labels", usá-lo; caso contrário, "temas"
    labs = d.get("labels", d.get("temas", []))
    # Junta os labels em uma única string e limpa
    labs_text = clean_text(" ".join(labs))
    labels_text.append(labs_text)

# Criar os tokenizadores com um token para palavras fora do vocabulário (<UNK>)
label_tokenizer = Tokenizer(oov_token="<UNK>")
label_tokenizer.fit_on_texts(labels_text)

# Para as descrições, usamos filters='' para não remover nenhum caractere e preservar os tokens especiais
description_tokenizer = Tokenizer(oov_token="<UNK>", filters='')
description_tokenizer.fit_on_texts(descriptions)

# Converter os textos em sequências numéricas
label_seq = label_tokenizer.texts_to_sequences(labels_text)
desc_seq = description_tokenizer.texts_to_sequences(descriptions)

# Definir comprimentos máximos:
# Limitamos o tamanho para evitar sequências muito longas (ex.: 20 para labels, 50 para descrições)
max_label_length = min(max(len(seq) for seq in label_seq), 20)
max_desc_length = min(max(len(seq) for seq in desc_seq), 50)

# Aplicar padding para padronizar o tamanho das sequências
label_padded = pad_sequences(label_seq, maxlen=max_label_length, padding='post')
desc_seq_padded = pad_sequences(desc_seq, maxlen=max_desc_length, padding='post')

# Dividir os dados em conjuntos de treino e teste (80% treino e 20% teste)
label_train, label_test, desc_train, desc_test = train_test_split(
    label_padded, desc_seq_padded, test_size=0.2, random_state=42
)

# Salvar os dados pré-processados (em formato .npy)
np.save('./models/label_train.npy', label_train)
np.save('./models/label_test.npy', label_test)
np.save('./models/desc_train.npy', desc_train)
np.save('./models/desc_test.npy', desc_test)

# Salvar os tokenizadores em arquivos .pkl para uso futuro
with open('./models/label_tokenizer.pkl', 'wb') as handle:
    pickle.dump(label_tokenizer, handle)
with open('./models/description_tokenizer.pkl', 'wb') as handle:
    pickle.dump(description_tokenizer, handle)

# Salvar os parâmetros de pré-processamento (comprimentos máximos)
preprocess_params = {'max_label_length': max_label_length, 'max_desc_length': max_desc_length}
with open('./models/preprocess_params.pkl', 'wb') as handle:
    pickle.dump(preprocess_params, handle)

print("Pré-processamento aprimorado concluído e dados salvos.")
