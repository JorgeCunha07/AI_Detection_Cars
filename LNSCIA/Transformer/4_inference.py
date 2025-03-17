import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Carregar o modelo Transformer treinado
model = load_model('./models/transformer_model.keras')

# Carregar tokenizadores e parâmetros
with open('./models/label_tokenizer.pkl', 'rb') as f:
    tokenizer_labels = pickle.load(f)
with open('./models/description_tokenizer.pkl', 'rb') as f:
    tokenizer_desc = pickle.load(f)
with open('./models/preprocess_params.pkl', 'rb') as f:
    preprocess_params = pickle.load(f)
    max_label_length = preprocess_params['max_label_length']
    max_desc_length = preprocess_params['max_desc_length']

index_word = tokenizer_desc.index_word

def generate_text_transformer(labels_input):
    # Converte a lista de labels em uma única string
    input_text = ", ".join(labels_input)
    # Converter labels para sequência numérica e padronizar
    labels_seq = tokenizer_labels.texts_to_sequences([input_text])
    labels_seq_padded = pad_sequences(labels_seq, maxlen=max_label_length, padding='post')
    labels_seq_padded = np.array(labels_seq_padded[:1])
    
    # Tokens especiais
    start_token = tokenizer_desc.word_index['startseq']
    end_token = tokenizer_desc.word_index['endseq']
    
    decoder_input = np.zeros((1, max_desc_length))
    decoder_input[0, 0] = start_token
    
    generated_tokens = []
    
    # Geração palavra a palavra (decodificação greedy)
    for i in range(1, max_desc_length):
        preds = model.predict([labels_seq_padded, decoder_input])
        token_index = np.argmax(preds[0, i-1, :])
        if token_index == end_token or token_index == 0:
            break
        generated_tokens.append(token_index)
        decoder_input[0, i] = token_index
        
    generated_text = ' '.join([index_word.get(tok, '') for tok in generated_tokens])
    return generated_text

# Exemplo de inferência
labels_test = ["chuva", "estrada movimentada", "semaforo vermelho", "peao", "passadeira"]
generated_description = generate_text_transformer(labels_test)
print(generated_description)
