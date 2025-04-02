# prompting.py
def build_prompt(user_input: str, conversation_history: list = None) -> str:
    """
    Constrói o prompt combinando instruções base e o histórico da conversa.
    """
    base_instructions = (
        "Instruções: Responda sempre em português de Portugal, usando termos e expressões comuns em Portugal.\n"
        "Mantenha um tom profissional mas amigável. Use termos específicos de Portugal, como 'peço desculpa', 'obrigado', etc.\n"
    )
    if conversation_history:
        history_text = "\n".join(conversation_history)
    else:
        history_text = f"Usuário: {user_input}"
    prompt = f"{base_instructions}{history_text}\nAssistente:"
    return prompt
