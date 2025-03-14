import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Lambda
import tensorflow.keras.backend as K

def build_seq2seq_model(encoder_vocab_size, decoder_vocab_size, max_decoder_seq_length, embedding_dim=64, latent_dim=128):
    # --- Encoder ---
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    encoder_embedding = Embedding(input_dim=encoder_vocab_size, output_dim=embedding_dim, mask_zero=True, name="encoder_embedding")(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    
    # --- Decoder ---
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_embedding_layer = Embedding(input_dim=decoder_vocab_size, output_dim=embedding_dim, mask_zero=True, name="decoder_embedding")
    decoder_embedded = decoder_embedding_layer(decoder_inputs)
    
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_dense = Dense(decoder_vocab_size, activation='softmax', name="decoder_dense")
    
    # --- Camada de Atenção ---
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, latent_dim, **kwargs):
            self.latent_dim = latent_dim
            super(AttentionLayer, self).__init__(**kwargs)
        
        def build(self, input_shape):
            # input_shape: [encoder_outputs, decoder_hidden_state]
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
            # Expande o estado do decoder para somar com cada time step do encoder
            decoder_hidden_state_expanded = K.expand_dims(decoder_hidden_state, 1)  # (batch, 1, latent_dim)
            # Calcula o score de atenção
            score = K.tanh(K.dot(encoder_outputs, self.Wa) + K.dot(decoder_hidden_state_expanded, self.Ua))
            attention_weights = K.softmax(K.dot(score, self.Va), axis=1)  # (batch, T_enc, 1)
            # Calcula o vetor de contexto como soma ponderada dos estados do encoder
            context_vector = attention_weights * encoder_outputs
            context_vector = K.sum(context_vector, axis=1)  # (batch, latent_dim)
            return context_vector, attention_weights

    attention_layer = AttentionLayer(latent_dim, name="attention_layer")
    
    # --- Loop do Decoder com Teacher Forcing ---
    all_outputs = []
    # Expande a dimensão do primeiro token (<start>) do embedding do decoder usando Lambda
    decoder_input = Lambda(lambda x: tf.expand_dims(x[:, 0, :], 1))(decoder_embedded)
    decoder_state_h, decoder_state_c = state_h, state_c

    for t in range(1, max_decoder_seq_length):
        # Executa um passo do LSTM do decoder
        decoder_output, decoder_state_h, decoder_state_c = decoder_lstm(decoder_input, initial_state=[decoder_state_h, decoder_state_c])
        
        # Calcula o vetor de contexto com o mecanismo de atenção
        context_vector, attn_weights = attention_layer([encoder_outputs, decoder_state_h])
        
        # Remove a dimensão extra do output do LSTM para concatenar com o contexto
        decoder_output_squeezed = Lambda(lambda x: tf.squeeze(x, axis=1))(decoder_output)
        
        # Concatena o output do LSTM com o vetor de contexto
        decoder_combined_context = Concatenate(axis=-1)([decoder_output_squeezed, context_vector])
        
        # Gera a predição do token seguinte
        output = decoder_dense(decoder_combined_context)
        output = Lambda(lambda x: tf.expand_dims(x, 1))(output)
        all_outputs.append(output)
        
        # Teacher forcing: utiliza o token real (embedding) para o próximo time step
        decoder_input = Lambda(lambda x, t=t: tf.expand_dims(x[:, t, :], 1))(decoder_embedded)
    
    # Utiliza a camada Concatenate para juntar as saídas ao longo do tempo
    decoder_outputs = Concatenate(axis=1)(all_outputs)
    
    # Modelo final: recebe as entradas do encoder e do decoder e gera a sequência de saída
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

if __name__ == "__main__":
    # Teste com parâmetros fictícios (verifique os valores com seu dataset real)
    model = build_seq2seq_model(encoder_vocab_size=7, decoder_vocab_size=24, max_decoder_seq_length=17, embedding_dim=64, latent_dim=128)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.summary()
