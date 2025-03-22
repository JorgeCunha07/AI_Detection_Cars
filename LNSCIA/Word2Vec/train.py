import json
import string
import os
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from unidecode import unidecode

# Baixar recursos do NLTK se necessário
nltk.download('punkt')

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def preprocess_text(text):
    """Normaliza o texto: converte para minúsculas, remove acentuação e pontuação, e faz a tokenização."""
    text = text.lower()
    text = unidecode(text)  # Remove acentos
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = word_tokenize(text, language='portuguese')
    return tokens

def load_questions_data(json_filename):
    """Carrega o dataset de perguntas/respostas a partir de um ficheiro JSON."""
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_corpus_from_questions(data):
    """Constrói um corpus a partir das perguntas e respostas do dataset JSON."""
    corpus = []
    for item in data:
        # Adiciona a pergunta
        question = item.get('pergunta', '')
        if question:
            corpus.append(preprocess_text(question))
        # Adiciona cada resposta (removendo o marcador "(correta)")
        for resposta in item.get('respostas', []):
            resposta_clean = resposta.replace("(correta)", "").strip()
            if resposta_clean:
                corpus.append(preprocess_text(resposta_clean))
    return corpus

def build_corpus_from_oscar(sample_size):
    """
    Carrega uma amostra do dataset OSCAR (Português) usando a biblioteca 'datasets'
    e constrói um corpus com sentenças processadas.
    """
    from datasets import load_dataset
    log("Carregando dataset OSCAR (Português)...")
    # O split 'train' e a versão "unshuffled_deduplicated_pt" fornecem textos em português
    dataset = load_dataset("oscar", "unshuffled_deduplicated_pt", split="train", streaming=True)
    
    corpus = []
    count = 0
    for example in dataset:
        text = example.get("text", "")
        if text:
            # Divide o texto em sentenças usando quebras de linha
            sentences = text.split("\n")
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    corpus.append(preprocess_text(sentence))
                    count += 1
                    if count >= sample_size:
                        log(f"Amostra do OSCAR atingida: {count} sentenças.")
                        return corpus
    log(f"Corpus OSCAR carregado com {count} sentenças.")
    return corpus

def train_word2vec(corpus, vector_size=300, window=7, min_count=5, workers=8, epochs=15, sg=1):
    log("Iniciando treinamento do modelo Word2Vec com os parâmetros ajustados...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=sg  # 1 para Skip-Gram; 0 para CBOW
    )
    log("Treinamento concluído!")
    return model

def main():
    # Carrega o corpus das perguntas/respostas
    json_filename = 'novas_perguntas_codigo_conducao.json'
    log("Carregando dados do JSON de perguntas...")
    questions_data = load_questions_data(json_filename)
    corpus_questions = build_corpus_from_questions(questions_data)
    log(f"Corpus de perguntas/respostas: {len(corpus_questions)} sentenças.")
    
    # Carrega uma amostra do OSCAR para português
    corpus_oscar = build_corpus_from_oscar(sample_size=10000)
    log(f"Corpus OSCAR: {len(corpus_oscar)} sentenças.")
    
    # Combina os dois corpora
    corpus_total = corpus_questions + corpus_oscar
    log(f"Corpus total utilizado para o treinamento: {len(corpus_total)} sentenças.")
    
    # Treina o modelo Word2Vec com o corpus combinado
    model = train_word2vec(corpus_total)
    
    # Salva o modelo treinado
    model_filename = "modelo_word2vec_melhorado.model"
    log(f"Salvando modelo em {model_filename}...")
    model.save(model_filename)
    log("Modelo salvo com sucesso!")

if __name__ == '__main__':
    main()
