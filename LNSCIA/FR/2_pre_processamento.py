import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Carregar os dados sintéticos
with open('synthetic_data_realistic.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extrair os rótulos e as descrições
# Para os rótulos, juntamos a lista em uma única string (para facilitar o tokenizing)
labels = [' '.join(d["labels"]) for d in data]
descriptions = [d["description"] for d in data]

# Criar e ajustar os tokenizadores
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

# No tokenizer das descrições, desabilitamos filtros para preservar os tokens especiais
description_tokenizer = Tokenizer(filters='')
description_tokenizer.fit_on_texts(descriptions)

# Converter textos em sequências
label_seq = label_tokenizer.texts_to_sequences(labels)
desc_seq = description_tokenizer.texts_to_sequences(descriptions)

# Calcular os comprimentos máximos das sequências
max_label_length = max(len(seq) for seq in label_seq)
max_desc_length = max(len(seq) for seq in desc_seq)

# Padronizar as sequências
label_padded = pad_sequences(label_seq, maxlen=max_label_length, padding='post')
desc_seq_padded = pad_sequences(desc_seq, maxlen=max_desc_length, padding='post')

# Dividir os dados em conjuntos de treino e teste
label_train, label_test, desc_train, desc_test = train_test_split(
    label_padded, desc_seq_padded, test_size=0.2, random_state=42
)

# Salvar as sequências padronizadas
np.save('label_train.npy', label_train)
np.save('label_test.npy', label_test)
np.save('desc_train.npy', desc_train)
np.save('desc_test.npy', desc_test)

# Salvar os tokenizadores para uso posterior
with open('label_tokenizer.pkl', 'wb') as handle:
    pickle.dump(label_tokenizer, handle)

with open('description_tokenizer.pkl', 'wb') as handle:
    pickle.dump(description_tokenizer, handle)

# Opcional: Salvar parâmetros de pré-processamento (comprimentos máximos)
preprocess_params = {'max_label_length': max_label_length, 'max_desc_length': max_desc_length}
with open('preprocess_params.pkl', 'wb') as handle:
    pickle.dump(preprocess_params, handle)

print("Pré-processamento concluído e dados guardados.")
