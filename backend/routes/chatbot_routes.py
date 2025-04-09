from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import random
from pathlib import Path
import json

from chatbot.validacao.validador_respostas import validar_resposta
from chatbot.pesquisa.similar_semantic import pesquisar_artigo
from chatbot.chat.inferencia import gerar_resposta_chat

router = APIRouter()

class ChatMessage(BaseModel):
    message: str

# Models para entrada de dados
class PerguntaRequest(BaseModel):
    pergunta: str


class RespostaQuizRequest(BaseModel):
    pergunta: str
    resposta: str


# Endpoint: Modo conversa (chat)
@router.post("/conversa")
async def handle_conversa(chat_input: ChatMessage):
    try:
        resposta = gerar_resposta_chat(chat_input.message)
        return {"response": resposta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint: Obter pergunta de quiz
@router.get("/quiz/pergunta")
def obter_pergunta():
    path_questoes = (
        Path(__file__).resolve().parent.parent / "chatbot" / "validacao" / "questoes.json"
    )
    with open(path_questoes, encoding="utf-8") as f:
        questoes = json.load(f)

    pergunta_data = random.choice(questoes)
    return {
        "pergunta": pergunta_data["pergunta"],
        "respostas_corretas": pergunta_data.get("respostas_corretas", []),
    }


# Endpoint: Validar resposta ao quiz
@router.post("/quiz/responder")
def responder_quiz(dados: RespostaQuizRequest):
    resultado = validar_resposta(dados.pergunta, dados.resposta)
    if resultado.get("correto"):
        return {"correto": True, "metodo": resultado.get("metodo", "desconhecido")}
    else:
        # Obtem respostas corretas para dar sugestão
        path_questoes = (
            Path(__file__).resolve().parent.parent / "chatbot" / "validacao" / "questoes.json"
        )
        with open(path_questoes, encoding="utf-8") as f:
            questoes = json.load(f)

        corretas = next(
            (
                q.get("respostas_corretas")
                for q in questoes
                if q["pergunta"] == dados.pergunta
            ),
            [],
        )
        sugestao = random.choice(corretas) if corretas else "Sem sugestão disponível"

        return {"correto": False, "sugestao": sugestao}


# Endpoint: Modo pesquisa (semântica)
@router.post("/pesquisa")
def pesquisar_info(req: PerguntaRequest):
    resultado = pesquisar_artigo(req.pergunta)
    return {
        "artigo_encontrado": resultado["artigo_encontrado"],
        "conteudo": resultado["conteudo"],
        "score_similaridade": resultado["score_similaridade"],
    }
