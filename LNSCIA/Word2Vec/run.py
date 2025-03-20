import json
import string
import numpy as np
import os
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from gensim.models import Word2Vec

# Baixar o recurso 'punkt' (se necessário)
nltk.download('punkt')

def log(message):
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

def load_model(model_filename):
    if os.path.exists(model_filename):
        log(f"Carregando modelo salvo: {model_filename}")
        model = Word2Vec.load(model_filename)
        log("Modelo carregado com sucesso!")
        return model
    else:
        log("Modelo não encontrado. Execute o script 'treinador.py' primeiro.")
        exit(1)

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
    if norm(vec1)==0 or norm(vec2)==0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def main():
    json_filename = 'perguntas_respostas_bomcondutor.json'
    model_filename = "modelo_word2vec.model"
    
    log("Carregando dados do JSON...")
    data = load_data(json_filename)
    log("Dados carregados com sucesso!")
    
    model = load_model(model_filename)
    
    threshold = 0.7
    log("Iniciando o diálogo com as perguntas:")
    for idx, item in enumerate(data):
        question = item.get('pergunta', '')
        log(f"Pergunta {idx+1}: {question}")
        
        # Seleciona a resposta correta (baseada na marcação "(correta)")
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
