import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# 1. Leitura do Corpus e Pré-processamento
# =============================================================================

# Abrir o ficheiro com o texto (certifique-se de que "input.txt" existe no diretório)
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

print("Tamanho do corpus (número de caracteres):", len(text))

# Utilizamos o Tokenizer para trabalhar a nível de palavras
tokenizer = Tokenizer()  # Usa por omissão split com espaços
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1  # Acrescenta 1 para o índice 0 (padding)
print("Número de palavras únicas:", vocab_size)

# =============================================================================
# 2. Criação das Sequências para Treino
# =============================================================================

# Definimos o comprimento da sequência de entrada (número de palavras)
sequence_length = 5  # Por exemplo, 5 palavras de entrada; o 6º será a palavra alvo

# Separamos o texto numa lista de palavras
words = text.split()

# Criamos as sequências: cada sequência terá 5 palavras (input) + 1 palavra (alvo)
sequences = []
for i in range(sequence_length, len(words)):
    seq = words[i-sequence_length:i+1]  # Obtém as 5 palavras anteriores e a palavra seguinte
    sequences.append(" ".join(seq))

print("Número de sequências criadas:", len(sequences))

# Convertendo as sequências para sequências de inteiros
sequences_int = tokenizer.texts_to_sequences(sequences)
sequences_int = pad_sequences(sequences_int, maxlen=sequence_length + 1, padding='pre')


# Separar em dados de entrada (X) e alvo (y)
X = sequences_int[:, :-1]   # As primeiras 5 palavras
y = sequences_int[:, -1]    # A 6ª palavra (alvo)

# Converter o vetor y para one-hot encoding (usando o vocabulário)
y = to_categorical(y, num_classes=vocab_size)

# Se quiser visualizar uma sequência exemplo:
print("Exemplo de sequência (texto):", sequences[0])
print("Exemplo de sequência (inteiros):", sequences_int[0])

# =============================================================================
# 3. Definição do Modelo de Geração de Texto com LSTM
# =============================================================================

# Definir as dimensões dos embeddings e o tamanho do estado da LSTM
embedding_dim = 100
latent_dim = 150

# Construir o modelo sequencial
model = Sequential()
# Camada de embedding que converte índices de palavras em vetores densos
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length))
# Camada LSTM que processa a sequência
model.add(LSTM(latent_dim))
# Camada densa com softmax para prever a palavra seguinte (entre vocab_size opções)
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# =============================================================================
# 4. Treinamento do Modelo
# =============================================================================

# Ajuste o número de épocas conforme necessário; para um projeto académico poderá ser útil
# aumentar o número de épocas ou ajustar o tamanho do batch
model.fit(X, y, batch_size=128, epochs=50)

# Opcional: guardar o modelo para utilização futura
model.save("text_generation_model.h5")
print("Modelo guardado em 'text_generation_model.h5'.")

# =============================================================================
# 5. Função para Gerar Texto a Partir de uma Semente
# =============================================================================

def gerar_texto(model, tokenizer, seed_text, num_palavras_gerar):
    """
    Função para gerar texto com base numa semente.
    Parâmetros:
      - model: o modelo treinado
      - tokenizer: o tokenizador utilizado no treino
      - seed_text: a semente de texto (por exemplo, "o sol nasce")
      - num_palavras_gerar: número de palavras a adicionar ao texto gerado
    Retorna:
      - Texto gerado (string)
    """
    resultado = seed_text
    for _ in range(num_palavras_gerar):
        # Converter a semente para uma sequência de inteiros
        token_list = tokenizer.texts_to_sequences([resultado])[0]
        # Fazer padding para garantir que a sequência tem o comprimento definido
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')
        # Prever a probabilidade da próxima palavra
        previsao = model.predict(token_list, verbose=0)
        # Selecionar o índice com maior probabilidade (decodificação greedy)
        indice_predito = np.argmax(previsao, axis=1)[0]
        # Procurar a palavra correspondente
        palavra_predita = ""
        for palavra, indice in tokenizer.word_index.items():
            if indice == indice_predito:
                palavra_predita = palavra
                break
        # Acrescentar a palavra prevista ao resultado
        resultado += " " + palavra_predita
    return resultado

# =============================================================================
# 6. Teste: Gerar Texto com Base em uma Semente de 4 ou 5 Palavras
# =============================================================================

# Exemplo de semente de 5 palavras
semente = "o sol nasce radiante"
num_palavras_para_gerar = 50  # Pode ajustar conforme desejado

texto_gerado = gerar_texto(model, tokenizer, semente, num_palavras_para_gerar)
print("\nSemente:", semente)
print("Texto gerado:")
print(texto_gerado)
