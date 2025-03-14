import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Camada de Atenção ---
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, latent_dim, **kwargs):
        self.latent_dim = latent_dim
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.Wa = self.add_weight(name='Wa',
                                  shape=(self.latent_dim, self.latent_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Ua = self.add_weight(name='Ua',
                                  shape=(self.latent_dim, self.latent_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Va = self.add_weight(name='Va',
                                  shape=(self.latent_dim, 1),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # inputs: [encoder_outputs, decoder_hidden_state]
        encoder_outputs, decoder_hidden_state = inputs  # encoder_outputs: (batch, T_enc, latent_dim)
                                                          # decoder_hidden_state: (batch, latent_dim)
        decoder_hidden_state_expanded = K.expand_dims(decoder_hidden_state, 1)  # (batch, 1, latent_dim)
        score = K.tanh(K.dot(encoder_outputs, self.Wa) + K.dot(decoder_hidden_state_expanded, self.Ua))
        attention_weights = K.softmax(K.dot(score, self.Va), axis=1)  # (batch, T_enc, 1)
        context_vector = attention_weights * encoder_outputs
        context_vector = K.sum(context_vector, axis=1)  # (batch, latent_dim)
        return context_vector, attention_weights

# --- Modelo do Encoder para Inferência ---
def build_encoder_inference(encoder_vocab_size, embedding_dim, latent_dim):
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    encoder_embedding = Embedding(input_dim=encoder_vocab_size, output_dim=embedding_dim,
                                  mask_zero=True, name="encoder_embedding")(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])
    return encoder_model

# --- Modelo do Decoder para Inferência ---
def build_decoder_inference(decoder_vocab_size, embedding_dim, latent_dim):
    decoder_inputs = Input(shape=(1,), name="decoder_input_inference")
    encoder_outputs_in = Input(shape=(None, latent_dim), name="encoder_outputs_inference")
    decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_input_c")
    
    decoder_embedding = Embedding(input_dim=decoder_vocab_size, output_dim=embedding_dim,
                                  mask_zero=True, name="decoder_embedding")
    decoder_embedded = decoder_embedding(decoder_inputs)  # (batch, 1, embedding_dim)
    
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedded, initial_state=[decoder_state_input_h, decoder_state_input_c])
    
    attn_layer = AttentionLayer(latent_dim, name="attention_layer")
    context_vector, attention_weights = attn_layer([encoder_outputs_in, state_h])
    
    decoder_output_squeezed = Lambda(lambda x: tf.squeeze(x, axis=1))(decoder_outputs)
    decoder_combined_context = Concatenate(axis=-1)([decoder_output_squeezed, context_vector])
    
    decoder_dense = Dense(decoder_vocab_size, activation='softmax', name="decoder_dense")
    decoder_outputs_final = decoder_dense(decoder_combined_context)
    
    decoder_model = Model(
        [decoder_inputs, encoder_outputs_in, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs_final, state_h, state_c, attention_weights]
    )
    return decoder_model

# --- Função de Decodificação (Decodificação Greedy) ---
def decode_sequence(input_seq, encoder_model, decoder_model, target_tokenizer, max_decoder_seq_length):
    # Codifica a sequência de entrada
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq)
    
    start_token_index = target_tokenizer.word_index['<start>']
    end_token_index = target_tokenizer.word_index['<end>']
    
    # Inicializa a sequência alvo com o token <start>
    target_seq = np.array([[start_token_index]])
    
    decoded_sentence = ''
    for _ in range(max_decoder_seq_length - 1):
        output_tokens, h, c, attn = decoder_model.predict([target_seq, encoder_outputs, state_h, state_c])
        sampled_token_index = np.argmax(output_tokens[0])
        
        # Recupera a palavra correspondente
        sampled_word = None
        for word, index in target_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        
        if sampled_word is None or sampled_word == '<end>':
            break
        
        decoded_sentence += ' ' + sampled_word
        target_seq = np.array([[sampled_token_index]])
        state_h, state_c = h, c
    
    return decoded_sentence.strip()

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Importe as variáveis do módulo de preparação de dados.
    # Certifique-se de que data_preparation.py exporta:
    # encoder_vocab_size, decoder_vocab_size, max_decoder_seq_length, max_encoder_seq_length, input_tokenizer, target_tokenizer
    from data_preparation import encoder_vocab_size as enc_vocab, decoder_vocab_size, max_decoder_seq_length, max_encoder_seq_length, input_tokenizer, target_tokenizer

    # **Importante:** Para carregar os pesos corretamente, o tamanho do vocabulário usado no encoder deve ser o mesmo que foi usado no treinamento.
    # Se os pesos salvos têm shape (24, 64) para o encoder_embedding, então:
    encoder_vocab_size = 24  # sobrescrevendo o valor importado se necessário

    embedding_dim = 64
    latent_dim = 128
    
    # Constrói os modelos de inferência
    encoder_model = build_encoder_inference(encoder_vocab_size, embedding_dim, latent_dim)
    decoder_model = build_decoder_inference(decoder_vocab_size, embedding_dim, latent_dim)
    
    # Carrega os pesos treinados (sem 'by_name')
    encoder_model.load_weights("outputs/model_weights/model.weights.h5")
    decoder_model.load_weights("outputs/model_weights/model.weights.h5")
    
    # Exemplo: Gere uma descrição para uma nova entrada de palavras-chave
    test_input = "peão passadeira semáforo dia chuva"
    test_seq = input_tokenizer.texts_to_sequences([test_input])
    test_seq = pad_sequences(test_seq, maxlen=max_encoder_seq_length, padding='post')
    
    decoded_sentence = decode_sequence(test_seq, encoder_model, decoder_model, target_tokenizer, max_decoder_seq_length)
    print("Descrição gerada:", decoded_sentence)
