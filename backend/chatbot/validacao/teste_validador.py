'''from scr.validacao.validador_respostas import validar_resposta

res = validar_resposta(
    "Qual a velocidade máxima permitida dentro das localidades para automóveis ligeiros sem reboque?",
    "Até 50 km/h",
    metodo_semantico="transformer"
)
print(res)

res = validar_resposta(
    "Qual a velocidade máxima permitida dentro das localidades para automóveis ligeiros sem reboque?",
    "Até 50 km/h"
)
print(res)

res = validar_resposta(
    "Qual a velocidade máxima permitida dentro das localidades para automóveis ligeiros sem reboque?",
    "Prai 40 quilómetros",
    metodo_semantico="transformer"
)
print(res)

res = validar_resposta(
    "Qual a velocidade máxima permitida dentro das localidades para automóveis ligeiros sem reboque?",
    "Mais 40 quilómetros"
)
print(res)'''

import json
from scr.validacao.validador_respostas import validar_resposta

# Carregar o banco de questões
with open("questoes.json", encoding="utf-8") as f:
    questoes = json.load(f)

# Exemplo de conjunto de testes (para cada questão, definir respostas "corretas" e "incorretas")
testes = [
    {
        "pergunta": "Qual a velocidade máxima permitida dentro das localidades para automóveis ligeiros sem reboque?",
        "resposta_correta": "Até 50 km/h",
        "resposta_incorreta": "40 km/h"
    },
    {
        "pergunta": "Qual a definição legal de 'autoestrada' segundo o Código da Estrada de Portugal?",
        "resposta_correta": "Uma via pública destinada a trânsito rápido, com separação física de faixas de rodagem e sem cruzamentos de nível.",
        "resposta_incorreta": "Qualquer estrada com mais de duas faixas."
    },
    # Adicione mais casos de teste conforme necessário...
]

acertos = 0
total = len(testes) * 2  # cada teste tem uma resposta correta e uma incorreta

for teste in testes:
    # Validação da resposta correta
    resultado = validar_resposta(teste["pergunta"], teste["resposta_correta"])
    if resultado.get("correto"):
        acertos += 1
    else:
        print(f"FALHA na questão: {teste['pergunta']} com resposta CORRETA.")

    # Validação da resposta incorreta
    resultado = validar_resposta(teste["pergunta"], teste["resposta_incorreta"])
    if not resultado.get("correto"):
        acertos += 1
    else:
        print(f"FALHA na questão: {teste['pergunta']} com resposta INCORRETA.")

print(f"Acurácia do validador: {acertos/total*100:.1f}%")
