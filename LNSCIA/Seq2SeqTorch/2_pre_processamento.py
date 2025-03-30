import os
import json
import re
import unicodedata
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Importar do nosso ficheiro tokenizer_utils
from tokenizer_utils import SimpleTokenizer, pad_sequences

import torch

# Função para limpar a pasta "./models/"
def clear_models_folder(folder_path='./models/'):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Não foi possível remover {file_path}. Motivo: {e}')
    else:
        os.makedirs(folder_path)

# Limpar a pasta "./models/" antes de iniciar
clear_models_folder('./models/')

# Função para limpar e normalizar o texto
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
with open('./data/descricoes_contexto_rodoviario.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

labels_text = []      # Para armazenar os rótulos (labels) em forma de string
descriptions = []     # Para armazenar as descrições

for d in data:
    # Processar a descrição: usa "description" ou, se não existir, "trecho"
    desc = d.get("description", d.get("trecho", ""))
    # Remover os tokens especiais temporariamente para limpeza
    desc = desc.replace("startseq", "").replace("endseq", "").strip()
    # Limpar e normalizar o texto
    desc_clean = clean_text(desc)
    # Re-adicionar os tokens especiais de início e fim
    desc_clean = "startseq " + desc_clean + " endseq"
    descriptions.append(desc_clean)
    
    # Processar os rótulos: usa "labels" ou "temas"
    labs = d.get("labels", d.get("temas", []))
    # Converte em string única e limpa
    labs_text_str = clean_text(" ".join(labs))
    labels_text.append(labs_text_str)

# Criar os tokenizadores usando o SimpleTokenizer
label_tokenizer = SimpleTokenizer(oov_token="<UNK>")
label_tokenizer.fit_on_texts(labels_text)

# Para o tokenizer de descrições, não aplicamos filtros (similar ao original)
description_tokenizer = SimpleTokenizer(oov_token="<UNK>", filters='')
description_tokenizer.fit_on_texts(descriptions)

# Converter os textos em sequências numéricas
label_seq = label_tokenizer.texts_to_sequences(labels_text)
desc_seq = description_tokenizer.texts_to_sequences(descriptions)

# Definir comprimentos máximos (limitando para evitar sequências muito longas)
max_label_length = min(max(len(seq) for seq in label_seq), 20)
max_desc_length = min(max(len(seq) for seq in desc_seq), 50)

# Aplicar padding para padronizar o tamanho das sequências
label_padded = pad_sequences(label_seq, maxlen=max_label_length, padding='post')
desc_padded = pad_sequences(desc_seq, maxlen=max_desc_length, padding='post')

# Dividir os dados em conjuntos de treino e teste (80% treino e 20% teste)
label_train, label_test, desc_train, desc_test = train_test_split(
    label_padded, desc_padded, test_size=0.2, random_state=42
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
preprocess_params = {
    'max_label_length': max_label_length,
    'max_desc_length': max_desc_length
}
with open('./models/preprocess_params.pkl', 'wb') as handle:
    pickle.dump(preprocess_params, handle)

print("Pré-processamento concluído e dados salvos.")

# Testar se o PyTorch detecta GPUs
print("PyTorch - GPUs detectadas:", torch.cuda.device_count())
