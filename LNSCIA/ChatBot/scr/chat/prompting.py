def criar_prompt(pergunta):
    prompt_base = (
        "Tu és um assistente que responde a perguntas sobre o Código da Estrada Português. "
        "Responde de forma clara e completa com base no código da estrada.\n"
        f"Pergunta: {pergunta}\nResposta:"
    )
    return prompt_base