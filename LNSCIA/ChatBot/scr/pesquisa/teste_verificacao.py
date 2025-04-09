
from similar_semantic import pesquisar_artigo

# Conjunto de queries com referência esperada
queries = [
    {
        "query": "O que diz o artigo 1.º do Código da Estrada?",
        "referencia_esperada": "Código da Estrada, Título I — Disposições gerais, Capítulo I — Princípios gerais, Artigo 1.º"
    },
    {
        "query": "Qual é a definição de autoestrada?",
        "referencia_esperada": "Código da Estrada, Título I — Disposições gerais, Capítulo I — Princípios gerais, Artigo 1.º"
    },
    {
        "query": "Quais são as penalidades por não obedecer às ordens das autoridades?",
        "referencia_esperada": "Código da Estrada, Título I — Disposições gerais, Capítulo I — Princípios gerais, Artigo 4.º"
    },
    # Adicione mais queries conforme necessário...
]

acertos = 0
for item in queries:
    resultado = pesquisar_artigo(item["query"])
    referencia = resultado.get("artigo_encontrado", "")
    score = resultado.get("score_similaridade", 0)
    print(f"Query: {item['query']}")
    print(f"Artigo Encontrado: {referencia}")
    print(f"Score: {score:.3f}")
    if item["referencia_esperada"].lower() in referencia.lower():
        acertos += 1
        print("Resultado: CORRETO\n")
    else:
        print("Resultado: INCORRETO\n")

print(f"Taxa de acerto da pesquisa semântica: {acertos/len(queries)*100:.1f}%")
