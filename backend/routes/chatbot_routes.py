from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import random
from pathlib import Path
import json

#from chatbot.validacao.validador_respostas import validar_resposta
#from chatbot.pesquisa.similar_semantic import pesquisar_artigo
#from chatbot.chat.inferencia import gerar_resposta_chat

router = APIRouter()

# Models
class Mensagem(BaseModel):
    pergunta: str

class RespostaQuiz(BaseModel):
    pergunta: str
    resposta: str

# Endpoints

@router.get("/conversa")
def conversa_endpoint():
    #resposta = gerar_resposta_chat(msg.pergunta)
    #return {"resposta": resposta}
    return "bom dia"

'''
@router.get("/quiz/pergunta")
def obter_pergunta():
    path_questoes = Path(__file__).resolve().parent.parent / "validacao" / "questoes.json"
    with open(path_questoes, encoding="utf-8") as f:
        questoes = json.load(f)
    pergunta_data = random.choice(questoes)
    return {
        "pergunta": pergunta_data["pergunta"],
        "respostas_corretas": pergunta_data.get("respostas_corretas", [])
    }


@router.post("/quiz/responder")
def responder_quiz(dados: RespostaQuiz):
    resultado = validar_resposta(dados.pergunta, dados.resposta)
    return resultado
'''

@router.post("/pesquisa")
def pesquisar_info(msg: Mensagem):
    resultado = pesquisar_artigo(msg.pergunta)
    return resultado
