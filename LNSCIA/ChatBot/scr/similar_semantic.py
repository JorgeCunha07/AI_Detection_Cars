# similar_semantic.py
from sentence_transformers import SentenceTransformer, util

# Inicializa o modelo para gerar embeddings semânticos.
# Esse modelo suporta múltiplos idiomas, incluindo português.
model_sem = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def similar_semantic(texto1, texto2, threshold=0.7):
    """
    Compara dois textos utilizando a similaridade do cosseno entre seus embeddings semânticos.
    Retorna True se a similaridade for maior ou igual ao threshold especificado.

    Parâmetros:
      texto1: str, primeiro texto a ser comparado.
      texto2: str, segundo texto a ser comparado.
      threshold: float, valor mínimo de similaridade para considerar os textos equivalentes.
    """
    # Gera embeddings para cada texto.
    embedding1 = model_sem.encode(texto1, convert_to_tensor=True)
    embedding2 = model_sem.encode(texto2, convert_to_tensor=True)
    
    # Calcula a similaridade do cosseno entre os embeddings.
    cosine_sim = util.pytorch_cos_sim(embedding1, embedding2)
    
    # Converte o resultado para float e compara com o threshold.
    return cosine_sim.item() >= threshold

if __name__ == '__main__':
    # Exemplo de uso
    resposta_usuario = "Não se ve a rua em pelo menos 50 metros"
    resposta_correta = "Quando o condutor não vê a faixa toda a largura em pelo menos 50 metros"
    
    if similar_semantic(resposta_usuario, resposta_correta):
        print("Respostas consideradas similares!")
    else:
        print("Respostas consideradas diferentes!")
