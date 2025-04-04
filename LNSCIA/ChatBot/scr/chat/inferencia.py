import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

# Diretório do modelo fine-tuned
model_dir = Path(__file__).resolve().parent.parent / "chat" / "gpt2-chat-finetuned"

# Carregar modelo e tokenizador
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def gerar_resposta_chat(user_input: str) -> str:
    # Prompt no formato "Usuário: [pergunta]\nAssistente:"
    prompt = f"Usuário: {user_input}\nAssistente:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,             # Geração determinística
            num_beams=5,                # Utiliza beam search para melhorar a qualidade
            top_p=0.92,
            top_k=50,
            num_return_sequences=1,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resposta = post_process_resposta(generated_text, prompt)
    return resposta

def post_process_resposta(generated_text: str, prompt: str) -> str:
    # Extrai a resposta removendo o prompt do texto gerado
    resposta = generated_text.replace(prompt, "").strip()

    if not resposta:
        resposta = "Desculpe, não consegui gerar uma resposta adequada. Pode reformular a pergunta?"
    if resposta.endswith("..."):
        resposta = "Desculpe, a resposta foi cortada. Pode repetir a pergunta de forma diferente?"
    if len(resposta.split()) < 5:
        resposta = "Desculpe, não consegui entender bem. Pode reformular a pergunta?"
    if resposta and not resposta.endswith(('.', '?', '!')):
        resposta += "."
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