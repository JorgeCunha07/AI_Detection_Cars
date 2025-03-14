import pandas as pd
import numpy as np
import math
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# 1. Leitura do dataset CSV
# -----------------------------
csv_filename = "dataset_exemplos.csv"  # Certifique-se de que este arquivo está no mesmo diretório
df = pd.read_csv(csv_filename, encoding='utf-8')

# Converte as colunas em listas
input_texts = df['vc_output'].astype(str).tolist()
target_texts = df['descricao'].astype(str).tolist()

print("Exemplos do dataset:")
for i in range(len(input_texts)):
    print(f"{input_texts[i]}  -->  {target_texts[i]}")

# -----------------------------
# 2. Criação dos Tokenizadores
# -----------------------------
# Para a entrada (output do VC), usamos lowercase e nenhum filtro
encoder_tokenizer = Tokenizer(lower=True, filters='')
encoder_tokenizer.fit_on_texts(input_texts)

# Para a saída, adicionamos tokens especiais <GO> e <EOS>
target_texts_mod = ["<GO> " + text + " <EOS>" for text in target_texts]
decoder_tokenizer = Tokenizer(lower=True, filters='')
decoder_tokenizer.fit_on_texts(target_texts_mod)

encoder_vocab_size = len(encoder_tokenizer.word_index) + 1  # índice 0 reservado para padding
decoder_vocab_size = len(decoder_tokenizer.word_index) + 1

print("Tamanho do vocabulário de entrada:", encoder_vocab_size)
print("Tamanho do vocabulário de saída:", decoder_vocab_size)

# -----------------------------
# 3. Conversão dos textos em sequências e Padding
# -----------------------------
encoder_sequences = encoder_tokenizer.texts_to_sequences(input_texts)
max_encoder_seq_length = max(len(seq) for seq in encoder_sequences)
encoder_input_data = pad_sequences(encoder_sequences, maxlen=max_encoder_seq_length, padding='post')

decoder_sequences = decoder_tokenizer.texts_to_sequences(target_texts_mod)
max_decoder_seq_length = max(len(seq) for seq in decoder_sequences)
decoder_input_data = pad_sequences(decoder_sequences, maxlen=max_decoder_seq_length, padding='post')

# Dados alvo do decoder (deslocados para a esquerda)
decoder_target_data = np.zeros_like(decoder_input_data)
decoder_target_data[:, :-1] = decoder_input_data[:, 1:]
decoder_target_data[:, -1] = 0

print("Comprimento máximo da entrada:", max_encoder_seq_length)
print("Comprimento máximo da saída:", max_decoder_seq_length)

# -----------------------------
# 4. Definição do modelo Seq2Seq (Encoder-Decoder)
# -----------------------------
embedding_dim = 64
latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
enc_emb = Embedding(input_dim=encoder_vocab_size, output_dim=embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
dec_emb = Embedding(input_dim=decoder_vocab_size, output_dim=embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(decoder_vocab_size, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

# -----------------------------
# 5. Treinamento do modelo
# -----------------------------
# Atenção: este dataset é pequeno; para obter melhores resultados, use um dataset maior e treine por mais épocas.
# Aqui aumentamos o número de épocas para 300 para simular um treinamento mais robusto.
model.fit([encoder_input_data, decoder_input_data],
          np.expand_dims(decoder_target_data, -1),
          batch_size=2,
          epochs=300,
          validation_split=0.2)

# -----------------------------
# 6. Construção dos modelos de inferência
# -----------------------------
# Modelo Encoder para inferência
encoder_model = Model(encoder_inputs, encoder_states)

# Modelo Decoder para inferência
decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Reutiliza a camada de embedding do decoder
dec_emb2 = Embedding(input_dim=decoder_vocab_size, output_dim=embedding_dim, mask_zero=True)(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs2] + decoder_states2)

# Cria mapeamento reverso para o vocabulário de saída
reverse_decoder_index = {i: word for word, i in decoder_tokenizer.word_index.items()}

# -----------------------------
# 7. Função de inferência com decodificação greedy com verificação de repetição
# -----------------------------
def decode_sequence_greedy(input_seq, max_length=max_decoder_seq_length, repetition_threshold=3):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.array([[decoder_tokenizer.word_index['<go>']]])
    
    stop_condition = False
    decoded_sentence = []
    last_token = None
    repeat_count = 0
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_decoder_index.get(sampled_token_index, '')
        
        decoded_sentence.append(sampled_word)
        
        # Verifica repetição: se o mesmo token se repetir muitas vezes consecutivamente, interrompe
        if sampled_word == last_token:
            repeat_count += 1
        else:
            last_token = sampled_word
            repeat_count = 1
        if repeat_count >= repetition_threshold:
            break
        
        # Condição de parada: se gerar <eos> ou atingir comprimento máximo
        if sampled_word == "<eos>" or len(decoded_sentence) >= max_length:
            stop_condition = True
        
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]
        
    return " ".join(decoded_sentence).replace(" <eos>", "")

# -----------------------------
# 8. Função de inferência usando Beam Search
# -----------------------------
def decode_sequence_beam(input_seq, beam_width=3, max_length=max_decoder_seq_length):
    states_value = encoder_model.predict(input_seq)
    
    # Cada candidato: (sequence_tokens, score, states)
    start_token = decoder_tokenizer.word_index['<go>']
    beam = [([start_token], 0.0, states_value)]
    completed_candidates = []
    
    for _ in range(max_length):
        new_beam = []
        for seq, score, states in beam:
            if seq[-1] == decoder_tokenizer.word_index.get('<eos>'):
                completed_candidates.append((seq, score))
                continue
            target_seq = np.array([[seq[-1]]])
            output_tokens, h, c = decoder_model.predict([target_seq] + states)
            # Para cada token possível, expande o candidato
            for token_index in range(1, decoder_vocab_size):
                token_prob = output_tokens[0, -1, token_index]
                if token_prob > 0:
                    new_seq = seq + [token_index]
                    new_score = score + math.log(token_prob)
                    new_beam.append((new_seq, new_score, [h, c]))
        # Ordena os candidatos e mantém os melhores beam_width
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]
        # Se todos os candidatos já terminaram com <eos>, podemos parar
        if all(seq[-1] == decoder_tokenizer.word_index.get('<eos>') for seq, _, _ in beam):
            break

    # Escolhe o candidato com maior score dentre os completados; se nenhum estiver completo, pega o melhor do beam.
    if completed_candidates:
        best_seq = max(completed_candidates, key=lambda x: x[1])[0]
    else:
        best_seq = beam[0][0]
        
    # Converte a sequência de índices em palavras
    decoded_sentence = " ".join([reverse_decoder_index.get(idx, '') for idx in best_seq])
    # Remove token <go> e <eos>
    decoded_sentence = decoded_sentence.replace("<go> ", "").replace(" <eos>", "")
    return decoded_sentence

import math

def decode_sequence_beam_penalty(input_seq, beam_width=3, max_length=max_decoder_seq_length, rep_penalty=1.2):
    # Obtém os estados iniciais do encoder
    states_value = encoder_model.predict(input_seq)
    
    start_token = decoder_tokenizer.word_index['<go>']
    end_token = decoder_tokenizer.word_index.get('<eos>')
    
    # Cada candidato é uma tupla: (sequência de tokens, score acumulado, estados)
    beam = [([start_token], 0.0, states_value)]
    completed_candidates = []
    
    for _ in range(max_length):
        new_beam = []
        for seq, score, states in beam:
            # Se já terminou com <EOS>, adiciona para candidatos completos
            if seq[-1] == end_token:
                completed_candidates.append((seq, score))
                continue
            target_seq = np.array([[seq[-1]]])
            output_tokens, h, c = decoder_model.predict([target_seq] + states)
            # Para cada token possível, expande a hipótese
            for token_index in range(1, decoder_vocab_size):
                token_prob = output_tokens[0, -1, token_index]
                if token_prob > 0:
                    # Penaliza se o token já apareceu na sequência
                    repetition_count = seq.count(token_index)
                    penalty = rep_penalty ** repetition_count  # quanto maior a repetição, maior a penalização
                    new_score = score + math.log(token_prob) - math.log(penalty)
                    new_seq = seq + [token_index]
                    new_beam.append((new_seq, new_score, [h, c]))
        # Ordena e mantém apenas as melhores hipóteses
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]
        # Se todas as hipóteses terminaram com <EOS>, para o loop
        if all(seq[-1] == end_token for seq, _, _ in beam):
            break

    # Escolhe o melhor candidato entre os completos ou, se nenhum, o melhor do beam
    if completed_candidates:
        best_seq = max(completed_candidates, key=lambda x: x[1])[0]
    else:
        best_seq = beam[0][0]
        
    # Converte a sequência de índices em palavras e remove tokens especiais
    decoded_sentence = " ".join([reverse_decoder_index.get(idx, '') for idx in best_seq])
    decoded_sentence = decoded_sentence.replace("<go> ", "").replace(" <eos>", "")
    return decoded_sentence


# -----------------------------
# 9. Teste: Geração de texto com as duas abordagens
# -----------------------------
test_input = "veiculo peao passadeira nublado dia"
test_seq = encoder_tokenizer.texts_to_sequences([test_input])
test_seq = pad_sequences(test_seq, maxlen=max_encoder_seq_length, padding='post')

greedy_result = decode_sequence_greedy(test_seq)
beam_result = decode_sequence_beam(test_seq, beam_width=3)
beam_result_penalty = decode_sequence_beam_penalty(test_seq, beam_width=3, rep_penalty=1.2)

model.save("seq2seq_model.h5")

print("\nInput (simulado VC):", test_input)
print("Texto gerado (decodificação greedy):", greedy_result)
print("Texto gerado (beam search):", beam_result)
print("Texto gerado (beam search penalty):", beam_result_penalty)