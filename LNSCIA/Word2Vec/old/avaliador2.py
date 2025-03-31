import json
import string
import numpy as np
from sentence_transformers import SentenceTransformer, util
import nltk
import os
from datetime import datetime

# Baixar recursos do NLTK (se ainda não tiver)
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def log(message):
    """Função para exibir mensagens com timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def preprocess_text(text):
    """
    Pré-processa o texto:
    - Converte para minúsculas
    - Remove pontuação
    - Tokeniza e remove stopwords
    Retorna o texto pré-processado como string.
    """
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = word_tokenize(text, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def load_data(json_filename):
    """Carrega os dados do arquivo JSON."""
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    json_filename = 'perguntas_respostas_bomcondutor.json'
    log("Carregando dados do JSON...")
    data = load_data(json_filename)
    log("Dados carregados com sucesso!")
    
    # Carregar um modelo SentenceTransformer pré-treinado para múltiplos idiomas.
    # Esse modelo (por exemplo, 'paraphrase-multilingual-MiniLM-L12-v2') suporta português.
    log("Carregando modelo SentenceTransformer pré-treinado...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    log("Modelo carregado com sucesso!")
    
    # Define o limiar de similaridade para considerar a resposta como correta.
    threshold = 0.7
    
    log("Iniciando o diálogo com as perguntas:")
    for idx, item in enumerate(data):
        question = item.get('pergunta', '')
        log(f"Pergunta {idx+1}: {question}")
        
        # Identifica a resposta correta (aquela que contém "(correta)") ou, se não houver, usa a primeira resposta.
        correct_answer = None
        for resposta in item.get('respostas', []):
            if "(correta)" in resposta:
                correct_answer = resposta.replace("(correta)", "").strip()
                break
        if correct_answer is None and item.get('respostas'):
            correct_answer = item.get('respostas')[0].strip()
            
        log(f"Resposta esperada: {correct_answer}")
        
        user_answer = input("Sua resposta: ")
        
        # Pré-processa as respostas
        preprocessed_correct = preprocess_text(correct_answer)
        preprocessed_user = preprocess_text(user_answer)
        
        # Obter embeddings
        embedding_correct = model.encode(preprocessed_correct, convert_to_tensor=True)
        embedding_user = model.encode(preprocessed_user, convert_to_tensor=True)
        
        # Calcular similaridade cosseno
        cosine_sim = util.pytorch_cos_sim(embedding_user, embedding_correct).item()
        log(f"Similaridade com a resposta esperada: {cosine_sim:.2f}")
        
        if cosine_sim >= threshold:
            log("Resposta considerada CORRETA!\n")
        else:
            log("Resposta considerada INCORRETA!\n")
        
        # Se desejar processar somente uma pergunta para teste, descomente a linha abaixo.
        # break
    
    log("Diálogo concluído.")

if __name__ == '__main__':
    main()
