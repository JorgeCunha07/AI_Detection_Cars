import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# Modelo
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Caminho para o JSON
base_dir = Path(__file__).resolve().parent
artigos_path = base_dir / "Codigo_Estrada.json"

# Carregar os artigos da chave "articles"
with open(artigos_path, encoding="utf-8") as f:
    artigos_json = json.load(f)["articles"]

# Extrair textos e referências
textos = [artigo["text"] for artigo in artigos_json]
referencias = [artigo["reference"] for artigo in artigos_json]

# Pré-computar os embeddings dos textos
embeddings = model.encode(textos, convert_to_tensor=True)

def extrair_numero(texto):
    """
    Extrai e retorna o primeiro número encontrado no texto.
    Se nenhum número for encontrado, retorna None.
    """
    match = re.search(r'\d+', texto)
    if match:
        return int(match.group())
    return None

def pesquisar_artigo(pergunta):
    pergunta_normalizada = pergunta.lower().strip()
    # Tenta extrair um número da consulta (ex.: de "artigo 123", extrai 123)
    numero_query = extrair_numero(pergunta_normalizada)
    
    # Busca de match direto: percorre os artigos extraindo o número de cada campo "article"
    if numero_query is not None:
        for artigo in artigos_json:
            numero_artigo = extrair_numero(artigo["article"].lower())
            if numero_artigo == numero_query:
                return {
                    "pergunta": pergunta,
                    "artigo_encontrado": artigo["reference"],
                    "conteudo": artigo["text"],
                    "score_similaridade": 1.0,
                    "metodo": "match direto - normalizado"
                }
    
    # Se não encontrar match direto, utiliza a pesquisa semântica
    emb_pergunta = model.encode(pergunta, convert_to_tensor=True)
    similaridades = util.cos_sim(emb_pergunta, embeddings)[0]
    idx = int(similaridades.argmax())
    return {
        "pergunta": pergunta,
        "artigo_encontrado": referencias[idx],
        "conteudo": textos[idx],
        "score_similaridade": float(similaridades[idx]),
        "metodo": "semantico"
    }

# Exemplo de uso
#consulta = "artigo 123"
#resultado = pesquisar_artigo(consulta)
#print(resultado)