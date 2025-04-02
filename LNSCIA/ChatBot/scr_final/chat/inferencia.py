import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

# Caminho para o modelo fine-tuned
modelo_dir = Path(__file__).resolve().parent.parent / "chat" / "gpt2-chat-finetuned"

# Carregar modelo e tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(modelo_dir)
model = GPT2LMHeadModel.from_pretrained(modelo_dir)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def gerar_resposta_chat(user_input: str) -> str:
    prompt = f"Usuário: {user_input}\nAssistente:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resposta = generated_text.split("Assistente:")[-1].strip()
    return resposta

# Modo standalone de teste — só corre se for executado diretamente
if __name__ == "__main__":
    print("🧪 Modo de inferência ativado. Escreve uma pergunta para testar o modelo.")
    print("❌ Escreve 'sair' para encerrar o programa.")

    while True:
        user_input = input("❓ Pergunta: ").strip()
        if user_input.lower() in ["sair", "exit", "q", "quit"]:
            print("👋 A encerrar inferência. Até à próxima, pequeno cientista dos dados.")
            break

        resposta = gerar_resposta_chat(user_input)
        print(f"🤖 Resposta: {resposta}")

        # Guardar apenas se não for comando de saída
        try:
            with open("historico_respostas.txt", "a", encoding="utf-8") as f:
                f.write(f"\nUsuário: {user_input}\nAssistente: {resposta}\n")
        except Exception as e:
            print(f"⚠️ Erro ao guardar histórico: {e}")
