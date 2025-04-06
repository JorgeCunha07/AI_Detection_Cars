'''from scr.pesquisa.similar_semantic import pesquisar_artigo

perguntas = [
    # 🟢 Básicas
    "O que diz o artigo 1.º do Código da Estrada?",
    "Qual é a definição de autoestrada?",
    "O que é uma zona de coexistência?",
    "Quais são os utilizadores vulneráveis?",
    "O que significa faixa de rodagem?",

    # 🟡 Médias
    "Qual é a regra para atravessar passagens de nível?",
    "Quais são as condições para circular em autoestradas?",
    "Existe limite de velocidade para pesados?",
    "O que diz o Código da Estrada sobre uso de telemóvel?",
    "Como se define uma berma?",

    # 🔴 Difíceis
    "Quais são as sanções para peões que circulam fora dos locais próprios?",
    "O que acontece se um condutor desobedecer à polícia de trânsito?",
    "Quais os deveres do condutor em relação aos peões?",
    "O que é considerado via reservada a automóveis e motociclos?",
    "Que regras se aplicam à condução em zonas escolares?",

    # 🤡 Marretas
    "Se eu for uma bicicleta voadora, posso andar na autoestrada?",
    "Posso conduzir descalço no Natal?",
    "Onde é que diz que parar em segunda fila é crime contra a humanidade?",
    "Qual é o artigo sobre motas que fazem mais barulho que foguetes?"
]

for i, pergunta in enumerate(perguntas, start=1):
    resultado = pesquisar_artigo(pergunta)
    print(f"\n🔎 Teste {i}: {pergunta}")
    print(f"📚 Artigo encontrado: {resultado.get('artigo_encontrado', 'Nenhum')}")
    print(f"📖 Conteúdo: {resultado.get('conteudo', '')[:300]}...")
    print(f"🎯 Similaridade: {round(resultado.get('score_similaridade', 0), 3)}")'''


from scr.pesquisa.similar_semantic import pesquisar_artigo

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
