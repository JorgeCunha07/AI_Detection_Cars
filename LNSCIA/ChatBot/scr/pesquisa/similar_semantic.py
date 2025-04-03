# scr_final/verificacao/similar_semantic.py
import json
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

# Pré-computar os embeddings
embeddings = model.encode(textos, convert_to_tensor=True)

# Função de pesquisa
def pesquisar_artigo(pergunta):
    pergunta_lower = pergunta.lower().strip()
    
    # Tenta encontrar "artigo x" diretamente no campo "article"
    for i, artigo in enumerate(artigos_json):
        if artigo["article"].lower() in pergunta_lower:
            return {
                "pergunta": pergunta,
                "artigo_encontrado": artigo["reference"],
                "conteudo": artigo["text"],
                "score_similaridade": 1.0,
                "metodo": "match direto"
            }

    # Se falhar, usa pesquisa semântica
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
