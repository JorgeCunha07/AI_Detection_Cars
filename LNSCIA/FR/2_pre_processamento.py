import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Carregar o dataset combinado
with open('./data/synthetic_and_real_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extrair os textos para os rótulos e as descrições.
labels_text = []
descriptions = []

for d in data:
    # Para a descrição, se existir "description" usa-o, senão "trecho"
    desc = d.get("description", d.get("trecho", ""))
    descriptions.append(desc)
    
    # Para os rótulos, se existir "labels" usa-os, senão "temas"
    labs = d.get("labels", d.get("temas", []))
    labels_text.append(" ".join(labs))

# Criar e ajustar os tokenizadores
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels_text)

description_tokenizer = Tokenizer(filters='')
description_tokenizer.fit_on_texts(descriptions)

# Converter textos em sequências
label_seq = label_tokenizer.texts_to_sequences(labels_text)
desc_seq = description_tokenizer.texts_to_sequences(descriptions)

# Determinar os comprimentos máximos
max_label_length = max(len(seq) for seq in label_seq)
max_desc_length = max(len(seq) for seq in desc_seq)

# Padronizar as sequências
label_padded = pad_sequences(label_seq, maxlen=max_label_length, padding='post')
desc_seq_padded = pad_sequences(desc_seq, maxlen=max_desc_length, padding='post')

# Dividir os dados em treino e teste
label_train, label_test, desc_train, desc_test = train_test_split(
    label_padded, desc_seq_padded, test_size=0.2, random_state=42
)

# Salvar as sequências padronizadas
np.save('./models/label_train.npy', label_train)
np.save('./models/label_test.npy', label_test)
np.save('./models/desc_train.npy', desc_train)
np.save('./models/desc_test.npy', desc_test)

# Salvar os tokenizadores
with open('./models/label_tokenizer.pkl', 'wb') as handle:
    pickle.dump(label_tokenizer, handle)
with open('./models/description_tokenizer.pkl', 'wb') as handle:
    pickle.dump(description_tokenizer, handle)

# Salvar os parâmetros de pré-processamento
preprocess_params = {'max_label_length': max_label_length, 'max_desc_length': max_desc_length}
with open('./models/preprocess_params.pkl', 'wb') as handle:
    pickle.dump(preprocess_params, handle)

print("Pré-processamento concluído e dados combinados salvos.")
