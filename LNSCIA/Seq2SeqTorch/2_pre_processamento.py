import json
import pickle
import numpy as np
from tokenizer_utils import SimpleTokenizer, pad_sequences

''' Labels permitidas:
veiculo: ["carro", "carros", "autocarro"],
veiculos: ["autocarros", "camião", "camiões"],
pessoa: ["peão", "ciclista"],
pessoas: ["peões", "ciclistas"],
infraestrutura: ["passadeira", "semáforo"],
infraestruturas: ["passadeiras", semáforos],
sinais_transito: ["sinal de stop", "sinal de limite de velocidade", "sinal de passadeira"]
weather_conditions = ["céu limpo", "nublado", "chuva", "nevoeiro", "vento", "neve"]
time_of_day = ["amanhecer", "anoitecer", "dia", "noite"]
locations = ["zona residencial", "parque de estacionamento", "túnel", "cidade", "autoestrada"]
'''

# Parâmetros
MIN_FREQ = 2
MAX_LABEL_LENGTH = 8
MAX_DESC_LENGTH = 30
MODEL_DIR = './models/'
DATA_DIR = './data/'

# Carregar dataset
with open(f'{DATA_DIR}/frases_com_labels.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Separar labels e descrições
labels = [" ".join(entry['labels']) for entry in data]
descriptions = [entry['description'] for entry in data]

# Tokenizar
label_tokenizer = SimpleTokenizer(oov_token='<OOV>')
label_tokenizer.fit_on_texts(labels, MIN_FREQ)
label_sequences = label_tokenizer.texts_to_sequences(labels)

# Tokenizador com filtragem por frequência
description_tokenizer = SimpleTokenizer(oov_token='<OOV>')
description_tokenizer.fit_on_texts(descriptions, MIN_FREQ)
# Filtrar palavras com frequência < MIN_FREQ
filtered_word_index = {w: i for w, i in description_tokenizer.word_index.items() if
                       description_tokenizer.word_counts[w] >= MIN_FREQ}
filtered_index_word = {i: w for w, i in filtered_word_index.items()}
description_tokenizer.word_index = filtered_word_index
description_tokenizer.index_word = filtered_index_word

desc_sequences = description_tokenizer.texts_to_sequences(descriptions)

# Padding
label_padded = pad_sequences(label_sequences, maxlen=MAX_LABEL_LENGTH, padding='post')
desc_padded = pad_sequences(desc_sequences, maxlen=MAX_DESC_LENGTH, padding='post')

# Verificar formato
desc_padded = np.array(desc_padded)
print("Shape do desc_padded:", desc_padded.shape)

# Guardar
np.save(f'{MODEL_DIR}/label_train.npy', label_padded)
np.save(f'{MODEL_DIR}/desc_train.npy', desc_padded)

with open(f'{MODEL_DIR}/label_tokenizer.pkl', 'wb') as f:
    pickle.dump(label_tokenizer, f)

with open(f'{MODEL_DIR}/description_tokenizer.pkl', 'wb') as f:
    pickle.dump(description_tokenizer, f)

with open(f'{MODEL_DIR}/preprocess_params.pkl', 'wb') as f:
    pickle.dump({
        'max_label_length': MAX_LABEL_LENGTH,
        'max_desc_length': MAX_DESC_LENGTH,
        'min_freq': MIN_FREQ
    }, f)
