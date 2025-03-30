import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Verifica se há GPU disponível
if not torch.cuda.is_available():
    print("GPU não encontrada. Este script requer GPU para funcionar.")
    sys.exit(1)

# Caminho para o modelo treinado
MODEL_PATH = "./meu_modelo_ajustado"

# Carrega o tokenizer e o modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda")

def gerar_pergunta():
    # Exemplo: seleciona uma pergunta predefinida.
    # No cenário real, pode ser feita a seleção aleatória de um dataset de Q&A.
    pergunta = "Qual é o limite de álcool no sangue para condutores em regime probatório?"
    resposta_correta = "0,2 g/l"
    return pergunta, resposta_correta

def validar_resposta(pergunta, resposta_correta, resposta_usuario):
    # Cria um prompt de validação para que o modelo compare as respostas.
    prompt = (
        f"Pergunta: {pergunta}\n"
        f"Resposta correta: {resposta_correta}\n"
        f"Resposta do usuário: {resposta_usuario}\n"
        "A resposta está correta? Explique brevemente."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        num_return_sequences=1
    )
    validacao = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return validacao

def main():
    # Geração da pergunta e obtenção da resposta correta
    pergunta, resposta_correta = gerar_pergunta()
    print("Pergunta:", pergunta)
    resposta_usuario = input("Sua resposta: ")
    
    # Validação da resposta
    validacao = validar_resposta(pergunta, resposta_correta, resposta_usuario)
    print("\nValidação:")
    print(validacao)

if __name__ == "__main__":
    main()
