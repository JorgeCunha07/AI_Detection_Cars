import json
import string
import os
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
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

def build_corpus(data):
    corpus = []
    for item in data:
        # Adiciona a pergunta
        question = item.get('pergunta', '')
        if question:
            corpus.append(preprocess_text(question))
        # Adiciona todas as respostas (removendo o marcador "(correta)")
        for resposta in item.get('respostas', []):
            resposta_clean = resposta.replace("(correta)", "").strip()
            if resposta_clean:
                corpus.append(preprocess_text(resposta_clean))
    return corpus

def train_word2vec(corpus, vector_size=300, window=7, min_count=3, workers=8, epochs=15, sg=1):
    log("Iniciando treinamento do modelo Word2Vec com parâmetros ajustados...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1  # 1 para Skip-Gram; 0 para CBOW
    )
    log("Treinamento concluído!")
    return model


def main():
    json_filename = 'perguntas_respostas_bomcondutor.json'
    log("Carregando dados do JSON...")
    data = load_data(json_filename)
    log("Dados carregados com sucesso!")
    
    log("Construindo corpus a partir dos dados completos...")
    corpus = build_corpus(data)
    log(f"Corpus construído com {len(corpus)} sentenças.")
    
    # Treinar o modelo Word2Vec com todo o dataset
    model = train_word2vec(corpus)
    
    # Salvar o modelo treinado
    model_filename = "modelo_word2vec.model"
    log(f"Salvando modelo em {model_filename}...")
    model.save(model_filename)
    log("Modelo salvo com sucesso!")

if __name__ == '__main__':
    main()
