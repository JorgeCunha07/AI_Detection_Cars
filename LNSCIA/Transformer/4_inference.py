import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

# Definição da camada customizada de Positional Encoding
class AddPositionalEncoding(Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super(AddPositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = self.compute_positional_encoding(max_len, d_model)

    def compute_positional_encoding(self, seq_len, d_model):
        positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        dims = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = positions * angle_rates
        # Para índices pares usa seno, para ímpares cosseno
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # Formato: (1, seq_len, d_model)
        return pos_encoding

    def call(self, x):
        seq_len = tf.shape(x)[1]
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        return x + pos_encoding

    def get_config(self):
        config = super(AddPositionalEncoding, self).get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model,
        })
        return config

# Carregar o modelo Transformer treinado,
# informando o objeto customizado necessário para a camada AddPositionalEncoding
model = load_model('./models/transformer_model.keras',
                   custom_objects={'AddPositionalEncoding': AddPositionalEncoding})

# Carregar tokenizadores e parâmetros de pré-processamento
with open('./models/label_tokenizer.pkl', 'rb') as f:
    tokenizer_labels = pickle.load(f)
with open('./models/description_tokenizer.pkl', 'rb') as f:
    tokenizer_desc = pickle.load(f)
with open('./models/preprocess_params.pkl', 'rb') as f:
    preprocess_params = pickle.load(f)
    max_label_length = preprocess_params['max_label_length']
    max_desc_length = preprocess_params['max_desc_length']

# Mapeamento do índice para palavra no tokenizador de descrições
index_word = tokenizer_desc.index_word

def generate_text_transformer(labels_input):
    # Converte a lista de labels em uma única string
    input_text = ", ".join(labels_input)
    # Converter os labels para sequência numérica e aplicar padding
    labels_seq = tokenizer_labels.texts_to_sequences([input_text])
    labels_seq_padded = pad_sequences(labels_seq, maxlen=max_label_length, padding='post')
    labels_seq_padded = np.array(labels_seq_padded[:1])
    
    # Tokens especiais do tokenizador de descrições
    start_token = tokenizer_desc.word_index.get('startseq')
    end_token = tokenizer_desc.word_index.get('endseq')
    
    # Inicializa a sequência do decoder com o token de início
    decoder_input = np.zeros((1, max_desc_length))
    decoder_input[0, 0] = start_token
    
    generated_tokens = []
    
    # Geração palavra a palavra (decodificação greedy)
    for i in range(1, max_desc_length):
        preds = model.predict([labels_seq_padded, decoder_input])
        # Obtem a predição para o último token gerado (posição i-1)
        token_index = np.argmax(preds[0, i-1, :])
        if token_index == end_token or token_index == 0:
            break
        generated_tokens.append(token_index)
        decoder_input[0, i] = token_index
        
    # Converte os tokens gerados em palavras
    generated_text = ' '.join([index_word.get(tok, '') for tok in generated_tokens])
    return generated_text

# Exemplo de inferência
if __name__ == '__main__':
    labels_test = ["chuva", "estrada movimentada", "semaforo vermelho", "peao", "passadeira"]
    generated_description = generate_text_transformer(labels_test)
    print("Descrição gerada:", generated_description)
