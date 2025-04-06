import gzip
import numpy as np
import pickle

FASTTEXT_PATH = "./data/cc.pt.300.vec.gz"
TOKENIZER_PATH = "./models/description_tokenizer.pkl"
OUTPUT_PATH = "./models/fasttext_embeddings.npy"

EMB_DIM = 300

# Carregar tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

vocab = tokenizer.word_index
vocab_size = max(vocab.values()) + 1
print(f"Vocab size: {vocab_size}")

# Inicializar matriz de embeddings
embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, EMB_DIM))
found = 0

# Ler vetores do ficheiro .gz
with gzip.open(FASTTEXT_PATH, "rt", encoding="utf-8") as f:
    next(f)  # Ignorar cabeçalho
    for line in f:
        values = line.rstrip().split(' ')
        word = values[0]
        if word in vocab:
            idx = vocab[word]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_matrix[idx] = vector
            found += 1

print(f"Encontrados {found} vetores de {len(vocab)} palavras do vocabulário.")

# Guardar matriz de embeddings
np.save(OUTPUT_PATH, embedding_matrix)
print(f"Matriz de embeddings salva em {OUTPUT_PATH}")
