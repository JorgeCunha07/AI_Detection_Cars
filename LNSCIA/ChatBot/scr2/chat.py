import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Carregar o modelo fine-tuned (certifique-se de que o diretório está correto)
model_name = "./gpt2-finetuned-wikipedia-pt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def conversar(prompt, max_length=512):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Geração com amostragem para respostas mais variadas
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    resposta = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return resposta

print("Digite 'sair' para encerrar a conversa.")
while True:
    usuario = input("Você: ")
    if usuario.lower() == "sair":
        break
    resposta_modelo = conversar(usuario)
    print("Modelo:", resposta_modelo)
