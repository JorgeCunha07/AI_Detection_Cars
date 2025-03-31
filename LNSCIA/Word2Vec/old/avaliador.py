import json
import string
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
import nltk
import os
from datetime import datetime

# Baixar o pacote 'punkt' se ainda não estiver disponível
nltk.download('punkt')

def log(message):
    # Exibe a mensagem com timestamp
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def preprocess_text(text):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = word_tokenize(text, language='portuguese')
    return tokens

def load_data(json_filename):
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_corpus(data):
    corpus = []
    for item in data:
        # Adiciona o texto da pergunta
        question = item.get('pergunta', '')
        if question:
            corpus.append(preprocess_text(question))
        # Adiciona os textos de todas as respostas (removendo o marcador "(correta)")
        for resposta in item.get('respostas', []):
            resposta_clean = resposta.replace("(correta)", "").strip()
            if resposta_clean:
                corpus.append(preprocess_text(resposta_clean))
    return corpus

def train_word2vec(corpus, vector_size=1000, window=5, min_count=5, workers=8, epochs=100):
    log("Iniciando treinamento do modelo Word2Vec com o dataset completo...")
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)
    log("Treinamento concluído!")
    return model

def load_or_train_model(corpus, model_filename="modelo_word2vec.model"):
    if os.path.exists(model_filename):
        log(f"Carregando modelo salvo: {model_filename}")
        model = Word2Vec.load(model_filename)
        log("Modelo carregado com sucesso!")
    else:
        model = train_word2vec(corpus)
        log(f"Salvando modelo em {model_filename}...")
        model.save(model_filename)
        log("Modelo salvo com sucesso!")
    return model

def sentence_vector(sentence, model):
    tokens = preprocess_text(sentence)
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def cosine_similarity(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def main():
    json_filename = 'perguntas_respostas_bomcondutor.json'
    log("Carregando dados do JSON...")
    data = load_data(json_filename)
    log("Dados carregados com sucesso!")
    
    log("Construindo corpus a partir dos dados completos...")
    corpus = build_corpus(data)
    log(f"Corpus construído com {len(corpus)} sentenças.")
    
    # Treinar ou carregar o modelo Word2Vec com o corpus completo
    model = load_or_train_model(corpus)
    
    # Define o limiar de similaridade para considerar a resposta como correta
    threshold = 0.7
    
    log("Iniciando o diálogo com todas as perguntas:")
    for idx, item in enumerate(data):
        question = item.get('pergunta', '')
        log(f"Pergunta {idx+1}: {question}")
        
        # Identifica a resposta correta (aquela que contém "(correta)")
        correct_answer = None
        for resposta in item.get('respostas', []):
            if "(correta)" in resposta:
                correct_answer = resposta.replace("(correta)", "").strip()
                break
        if correct_answer is None and item.get('respostas'):
            correct_answer = item.get('respostas')[0].strip()
            
        log(f"Resposta esperada: {correct_answer}")
        user_answer = input("Sua resposta: ")
        
        user_vec = sentence_vector(user_answer, model)
        correct_vec = sentence_vector(correct_answer, model)
        similarity = cosine_similarity(user_vec, correct_vec)
        log(f"Similaridade com a resposta esperada: {similarity:.2f}")
        
        if similarity >= threshold:
            log("Resposta considerada CORRETA!\n")
        else:
            log("Resposta considerada INCORRETA!\n")
        
    log("Diálogo concluído.")

if __name__ == '__main__':
    main()
