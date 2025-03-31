import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Carrega o modelo e o tokenizer treinados (assumindo que foram fine-tuned com 20 epochs)
model_dir = "./trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Define o dispositivo (GPU se disponível, caso contrário, CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate_answer(question, user_answer):
    """
    Monta o prompt com a pergunta e a resposta do utilizador e gera a avaliação.
    O prompt inclui a instrução:
      "Responda APENAS com 'Correto' ou 'Incorreto', sem nenhuma explicação adicional."
    """
    prompt = (
        f"Pergunta: {question}\n"
        f"Resposta: {user_answer}\n"
        "Avaliação: Responda APENAS com 'Correto' ou 'Incorreto', sem nenhuma explicação adicional."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + 3,  # Limita a apenas alguns tokens após o prompt
        do_sample=False,
        num_beams=5,
        early_stopping=True,
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extrai somente o texto gerado após o prompt
    eval_text = output_text[len(prompt):].strip()
    return eval_text

# Define a pergunta de teste
question = "Qual a definição legal de 'autoestrada' segundo o Código da Estrada de Portugal?"

# Lista de 10 respostas CORRETAS (5 originais + 5 adicionais)
correct_responses = [
    "Uma via pública destinada a trânsito rápido, com separação física de faixas de rodagem e sem cruzamentos de nível.",
    "Uma via rápida, sem acessos diretos a propriedades marginais e com acessos condicionados.",
    "Uma via especialmente sinalizada, sem cruzamentos, com separação física das faixas de rodagem.",
    "Uma via pública com separação física de faixas, destinada ao trânsito rápido e sinalizada adequadamente.",
    "Uma estrada pública destinada a tráfego rápido, sem acessos diretos e com sinalização própria.",
    "Uma autoestrada é uma via de alta velocidade, com separação física entre faixas e sem interseções de nível.",
    "Uma via destinada ao trânsito rápido, com acessos condicionados e sem cruzamentos a nível, caracteriza uma autoestrada.",
    "Uma autoestrada é definida como uma via pública para tráfego rápido, com divisões físicas claras e sem cruzamentos no mesmo nível.",
    "Uma via pública para trânsito rápido, com separação física de faixas e acessos controlados, é considerada autoestrada.",
    "Uma autoestrada é uma via de trânsito rápido, com faixas separadas fisicamente, sem cruzamentos de nível e com acessos condicionados."
]

# Lista de 10 respostas INCORRETAS (5 originais + 5 adicionais)
incorrect_responses = [
    "Uma estrada sem limite de velocidade.",
    "Qualquer estrada com mais de duas faixas é considerada autoestrada.",
    "Uma via urbana rápida sem sinalização específica é autoestrada.",
    "Uma estrada pública com acessos frequentes às propriedades marginais é autoestrada.",
    "Uma estrada onde são permitidos cruzamentos de nível pode ser considerada autoestrada.",
    "Uma autoestrada é uma estrada sem restrições, onde não há limites de velocidade.",
    "Qualquer via pública com múltiplas faixas é automaticamente considerada autoestrada.",
    "Uma via sem separação física, mas com alta velocidade, é classificada como autoestrada.",
    "Uma autoestrada permite cruzamentos de nível, desde que sejam sinalizados.",
    "Uma estrada com acessos diretos a propriedades marginais pode ser considerada autoestrada."
]

print("Testando respostas CORRETAS para a pergunta:")
print(question)
print("-" * 80)
for i, resp in enumerate(correct_responses):
    evaluation = evaluate_answer(question, resp)
    print(f"Resposta CORRETA {i+1}: {resp}")
    print("Avaliação:", evaluation)
    print("-" * 80)

print("\nTestando respostas INCORRETAS para a mesma pergunta:")
print(question)
print("-" * 80)
for i, resp in enumerate(incorrect_responses):
    evaluation = evaluate_answer(question, resp)
    print(f"Resposta INCORRETA {i+1}: {resp}")
    print("Avaliação:", evaluation)
    print("-" * 80)
