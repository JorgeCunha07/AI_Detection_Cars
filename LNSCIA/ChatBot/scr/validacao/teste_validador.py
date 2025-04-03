from scr.validacao.validador_respostas import validar_resposta

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
print(res)