import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import re

# Diretório do modelo fine-tuned
model_dir = Path(__file__).resolve().parent.parent / "chat" / "gpt2-chat-finetuned"

# Carregar modelo e tokenizador
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Palavras-chave relevantes para o tema
palavras_chave = [
    "carro", "conduzir", "condutor", "carta", "travar", "buzina", "direção", "aceleração", "travagem", "marcha-atrás",
    "sinal", "semáforo", "stop", "prioridade", "ceda passagem", "sinalização", "linha contínua", "triângulo", "pisca", "luzes", "máximos", "médios", "nevoeiro",
    "infrações", "multa", "coima", "pontos", "penalização", "alcoolemia", "teste do balão", "excesso de velocidade", "polícia", "fiscalização",
    "estacionar", "parar", "berma", "passeio", "segunda fila", "garagem", "estacionamento", "zona reservada", "local proibido",
    "cinto", "cadeira auto", "cadeirinha", "criança", "transporte", "animal", "cão", "gato", "segurança", "colete refletor", "triângulo de sinalização",
    "autoestrada", "estrada nacional", "rotunda", "faixa", "via", "passadeira", "cruzamento", "ciclovia", "acostamento", "zona escolar",
    "bicicleta", "ciclomotor", "mota", "trotinete", "trator", "reboque", "mercadorias",
    "pneu", "inspeção", "óleo", "motor", "avaria", "revisão", "vidro partido", "buzina avariada", "travões"
]

def pergunta_valida(pergunta: str) -> bool:
    pergunta_lower = pergunta.lower()
    return any(palavra in pergunta_lower for palavra in palavras_chave)

def post_process_resposta(generated_text: str, prompt: str) -> str:
    resposta = generated_text.replace(prompt, "").strip()

    # Remove frases redundantes e palavras a mais
    resposta = resposta.split(".")[0]  # Só fica com a primeira frase curta
    resposta = resposta.strip()

    if not resposta:
        resposta = "Desculpe, não consegui gerar uma resposta."
    elif len(resposta.split()) < 4:
        resposta = "Precisa de reformular a pergunta."
    else:
        resposta += "."

    return resposta

def gerar_resposta_chat(user_input: str) -> str:
    if not pergunta_valida(user_input):
        return "Sou um assistente especializado apenas em temas do Código da Estrada. Reformule a sua pergunta."

    # Prompt no formato "Usuário: [pergunta]\nAssistente:"
    prompt = f"Usuário: {user_input}\nAssistente: " 
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,             # Sem sampling para ser determinístico
            num_beams=3,                 # Pouco criativo, mas confiável
            temperature=0.7,             # Respostas menos malucas
            top_p=0.9,
            top_k=30,
            repetition_penalty=2.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
        )


    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resposta = post_process_resposta(generated_text, prompt)
    return resposta


if __name__ == "__main__":
    print("🧪 Modo de inferência ativado. Escreva uma pergunta para testar o modelo.")
    print("❌ Digite 'sair' para encerrar.")

    while True:
        user_input = input("❓ Pergunta: ").strip()
        if user_input.lower() in ["sair", "exit", "q", "quit"]:
            print("👋 Encerrando inferência. Até à próxima!")
            break

        resposta = gerar_resposta_chat(user_input)
        print(f"🤖 Resposta: {resposta}")