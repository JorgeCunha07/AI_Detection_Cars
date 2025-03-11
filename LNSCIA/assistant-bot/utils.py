import nltk
import pickle
import re
import numpy as np
import unicodedata  # Para normalização de caracteres acentuados
import spacy  # Para processamento de linguagem natural em português

# Carregar o modelo de português do spaCy
try:
    nlp = spacy.load('pt_core_news_sm')
except:
    # Se o modelo específico não estiver disponível, tente um modelo genérico
    try:
        nlp = spacy.load('pt')
    except:
        print("Aviso: Modelo spaCy para português não encontrado. Usando processamento básico.")
        nlp = None

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'data/word_embeddings.tsv',
}

# Dicionário de sinônimos em português para termos comuns de programação
# Mapeia termos em português para seus equivalentes em inglês
PROGRAMMING_SYNONYMS = {
    # Python
    'lista': 'list',
    'dicionario': 'dictionary',
    'dicionário': 'dictionary',
    'tupla': 'tuple',
    'conjunto': 'set',
    'classe': 'class',
    'funcao': 'function',
    'função': 'function',
    'metodo': 'method',
    'método': 'method',
    'variavel': 'variable',
    'variável': 'variable',
    'string': 'string',
    'inteiro': 'integer',
    'flutuante': 'float',
    'booleano': 'boolean',
    'verdadeiro': 'true',
    'falso': 'false',
    'nulo': 'null',
    'none': 'none',
    'importar': 'import',
    'imprimir': 'print',
    'entrada': 'input',
    'saida': 'output',
    'saída': 'output',
    'erro': 'error',
    'excecao': 'exception',
    'exceção': 'exception',
    'loop': 'loop',
    'laço': 'loop',
    'condicional': 'conditional',
    'se': 'if',
    'senao': 'else',
    'senão': 'else',
    'enquanto': 'while',
    'para': 'for',
    'retornar': 'return',
    'tentar': 'try',
    'exceto': 'except',
    'finalmente': 'finally',
    'arquivo': 'file',
    'abrir': 'open',
    'ler': 'read',
    'escrever': 'write',
    'fechar': 'close',
    
    # Web
    'html': 'html',
    'css': 'css',
    'javascript': 'javascript',
    'banco de dados': 'database',
    'sql': 'sql',
    'servidor': 'server',
    'cliente': 'client',
    'api': 'api',
    'requisicao': 'request',
    'requisição': 'request',
    'resposta': 'response',
    
    # Geral
    'codigo': 'code',
    'código': 'code',
    'programa': 'program',
    'programacao': 'programming',
    'programação': 'programming',
    'algoritmo': 'algorithm',
    'depuracao': 'debugging',
    'depuração': 'debugging',
    'compilar': 'compile',
    'executar': 'execute',
    'implementar': 'implement',
    'otimizar': 'optimize',
    'biblioteca': 'library',
    'framework': 'framework',
    'desenvolvimento': 'development',
    'teste': 'test',
    'versao': 'version',
    'versão': 'version',
    'documentacao': 'documentation',
    'documentação': 'documentation',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing for Portuguese text."""
    # Usar spaCy para processamento avançado se disponível
    if nlp is not None:
        # Processar o texto com spaCy
        doc = nlp(text.lower())
        
        # Extrair lemas e filtrar stopwords e pontuação
        processed_tokens = []
        for token in doc:
            # Verificar se não é stopword, pontuação ou espaço
            if not token.is_stop and not token.is_punct and not token.is_space:
                # Usar o lema (forma base da palavra)
                word = token.lemma_
                
                # Verificar se a palavra está no dicionário de sinônimos
                if word in PROGRAMMING_SYNONYMS:
                    word = PROGRAMMING_SYNONYMS[word]
                
                processed_tokens.append(word)
        
        return ' '.join(processed_tokens)
    
    else:
        # Processamento básico sem spaCy
        # Substituir caracteres especiais por espaço
        replace_by_space_re = re.compile('[/(){}\[\]\|@,;:!?]')
        
        # Expressão regular que mantém letras (incluindo acentuadas), números e alguns símbolos
        good_symbols_re = re.compile('[^0-9a-záàâãéèêíìóòôõúùûç #+_]')
        
        # Usar stopwords em português
        stopwords_set = set(stopwords.words('portuguese'))

        # Normalizar texto
        text = text.lower()
        text = replace_by_space_re.sub(' ', text)
        text = good_symbols_re.sub('', text)
        
        # Processar palavras e verificar sinônimos
        words = []
        for word in text.split():
            if word and word not in stopwords_set:
                # Verificar se a palavra está no dicionário de sinônimos
                if word in PROGRAMMING_SYNONYMS:
                    word = PROGRAMMING_SYNONYMS[word]
                words.append(word)
        
        return ' '.join(words)


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.
    Args:
      embeddings_path - path to the embeddings file.
    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # Hint: you have already implemented a similar routine previously.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    embeddings = {}
    for line in open(embeddings_path, encoding='utf-8'):
        w, *v = line.strip().split('\t')
        embeddings[w] = np.asfarray(v, dtype=np.float32)

    embeddings_dim = next(iter(embeddings.values())).size

    return embeddings, embeddings_dim
    

def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings.
    Improved to handle Portuguese words better by using spaCy and synonyms.
    """
    # Processar a pergunta com spaCy se disponível
    if nlp is not None:
        doc = nlp(question.lower())
        # Extrair lemas e filtrar stopwords e pontuação
        words = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                word = token.lemma_
                # Verificar se a palavra está no dicionário de sinônimos
                if word in PROGRAMMING_SYNONYMS:
                    word = PROGRAMMING_SYNONYMS[word]
                words.append(word)
    else:
        # Processamento básico sem spaCy
        words = question.lower().split()
        # Verificar sinônimos
        words = [PROGRAMMING_SYNONYMS.get(w, w) for w in words]
    
    # Calcular o vetor médio
    sum_vector = np.zeros(dim)
    n_vectors = 0
    
    for w in words:
        # Tenta encontrar a palavra nos embeddings
        if w in embeddings:
            sum_vector += embeddings[w]
            n_vectors += 1
        else:
            # Se não encontrar, tenta a versão sem acentos
            w_normalized = normalize_accents(w)
            if w_normalized in embeddings:
                sum_vector += embeddings[w_normalized]
                n_vectors += 1
            # Se ainda não encontrar, tenta versões em minúsculas
            elif w.lower() in embeddings:
                sum_vector += embeddings[w.lower()]
                n_vectors += 1
            elif w_normalized.lower() in embeddings:
                sum_vector += embeddings[w_normalized.lower()]
                n_vectors += 1
    
    qv_emb = sum_vector
    if n_vectors > 0:  # Evitar divisão por zero
        qv_emb = sum_vector / n_vectors

    return qv_emb


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def normalize_accents(text):
    """Remove acentos de um texto, mantendo a letra base.
    Por exemplo: 'olá' -> 'ola', 'café' -> 'cafe'
    """
    # Normaliza para forma NFD e remove os caracteres de combinação
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                  if not unicodedata.combining(c))
