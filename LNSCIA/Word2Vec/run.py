import json
import string
import numpy as np
import os
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from gensim.models import Word2Vec
from unidecode import unidecode
import difflib
import spacy

# Baixar recursos do NLTK (punkt e stopwords)
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('portuguese'))

# Inicializa spaCy para lematização
try:
    nlp = spacy.load("pt_core_news_sm")
except Exception as e:
    print("SpaCy model 'pt_core_news_sm' não encontrado. Por favor, instale-o com 'python -m spacy download pt_core_news_sm'.")
    exit(1)

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def preprocess_text(text):
    """
    Normaliza o texto: converte para minúsculas, remove acentos, pontuação, stopwords
    e realiza a lematização.
    """
    text = text.lower()
    text = unidecode(text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.text.strip() != '']
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
        log("Modelo não encontrado. Execute o script 'train.py' primeiro.")
        exit(1)

def sentence_vector(sentence, model):
    tokens = preprocess_text(sentence)
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def cosine_similarity(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def text_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

def token_synonym_match(user_tokens, correct_tokens, model, threshold=0.8):
    """
    Compara cada token da resposta do usuário com os tokens da resposta correta.
    Se a similaridade entre dois tokens for igual ou superior ao threshold,
    conta como um match.
    Retorna a proporção de tokens corretos que encontram um sinônimo na resposta do usuário.
    """
    matches = 0
    for ct in correct_tokens:
        for ut in user_tokens:
            if ut in model.wv and ct in model.wv:
                sim = cosine_similarity(model.wv[ut], model.wv[ct])
                if sim >= threshold:
                    matches += 1
                    break  # Considera apenas um match por token correto
    return matches / len(correct_tokens) if correct_tokens else 0

def main():
    json_filename = 'novas_perguntas_codigo_conducao.json'
    model_filename = "modelo_word2vec_melhorado.model"
    
    # Modo de teste: respostas simuladas para testes automáticos
    test_mode = True
    simulated_answers = [
        "para na berma da estrada.",
        "quando chuve muito.",
        "uma passadeira.",
        "0,2 g/l",
        "usar kit de mãos livres.",
        "quando nao existe visibilidade.",
        "saiu do carro.",
        "a 30 metros do carro.",
        "ignorar as regras.",
        "em paralelo com outro."
    ]
    
    log("Carregando dados do JSON...")
    data = load_data(json_filename)
    log("Dados carregados com sucesso!")
    
    model = load_model(model_filename)
    
    # Definição dos thresholds e pesos para as métricas
    cosine_threshold = 0.6
    text_threshold = 0.7
    combined_threshold = 0.7
    synonym_threshold = 0.8  # Para similaridade entre tokens individuais
    weight_cosine = 0.4
    weight_text = 0.3
    weight_synonym = 0.3
    
    log("Iniciando o diálogo com as perguntas:")
    for idx, item in enumerate(data):
        question = item.get('pergunta', '')
        log(f"Pergunta {idx+1}: {question}")
        
        # Seleciona as respostas corretas (marcadas com "(correta)")
        correct_answers = [resp.replace("(correta)", "").strip() 
                           for resp in item.get('respostas', []) if "(correta)" in resp]
        if not correct_answers and item.get('respostas'):
            correct_answers = [item.get('respostas')[0].strip()]
            
        log(f"Resposta(s) esperada(s): {correct_answers}")
        
        if test_mode:
            try:
                user_answer = simulated_answers[idx]
                log(f"Resposta simulada: {user_answer}")
            except IndexError:
                user_answer = ""
                log("Nenhuma resposta simulada definida para esta pergunta.")
        else:
            user_answer = input("Sua resposta: ")
        
        user_answer_norm = unidecode(user_answer.lower().strip())
        
        # Verificação de correspondência textual exata parcial
        match_found = False
        for correct_answer in correct_answers:
            correct_answer_norm = unidecode(correct_answer.lower().strip())
            if correct_answer_norm in user_answer_norm or user_answer_norm in correct_answer_norm:
                log("Correspondência textual exata encontrada. Resposta considerada CORRETA!\n")
                match_found = True
                break
        if match_found:
            continue
        
        # Avalia cada resposta correta e seleciona a melhor similaridade combinada
        best_combined = 0
        for correct_answer in correct_answers:
            user_vec = sentence_vector(user_answer, model)
            correct_vec = sentence_vector(correct_answer, model)
            sim_cos = cosine_similarity(user_vec, correct_vec)
            
            correct_answer_norm = unidecode(correct_answer.lower().strip())
            sim_text = text_similarity(user_answer_norm, correct_answer_norm)
            
            user_tokens = preprocess_text(user_answer)
            correct_tokens = preprocess_text(correct_answer)
            sim_synonym = token_synonym_match(user_tokens, correct_tokens, model, threshold=synonym_threshold)
            
            log(f"Similaridade coseno: {sim_cos:.2f}")
            log(f"Similaridade textual: {sim_text:.2f}")
            log(f"Proporcao de sinônimos encontrados: {sim_synonym:.2f}")
            
            combined_sim = weight_cosine * sim_cos + weight_text * sim_text + weight_synonym * sim_synonym
            log(f"Similaridade combinada: {combined_sim:.2f}")
            
            if combined_sim > best_combined:
                best_combined = combined_sim
        
        if best_combined >= combined_threshold:
            log("Resposta considerada CORRETA!\n")
        else:
            log("Resposta considerada INCORRETA!\n")
    
    log("Diálogo concluído.")

if __name__ == '__main__':
    main()
