import torch
import string
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Carrega o modelo fine-tuned e o tokenizer
model_dir = "./trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Configura o dispositivo (GPU se disponível, caso contrário, CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate_answer(question, user_answer):
    """
    Monta o prompt com a pergunta e a resposta do utilizador e gera a avaliação.
    O prompt inclui a instrução para que o modelo gere apenas "correto" ou "incorreto".
    """
    # Reformulação do prompt: a instrução fica em uma nova linha para maior clareza
    prompt = (
        f"Pergunta: {question}\n"
        f"Resposta: {user_answer}\n"
        'Avaliação: Responda APENAS com "Correto" OU "Incorreto".'
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,  # Permite gerar até 50 tokens novos
        do_sample=False,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extrai o texto gerado após o prompt
    eval_text = output_text[len(prompt):].strip()
    return eval_text

def normalize_output(output):
    """
    Normaliza a resposta extraindo o primeiro token, removendo espaços extras e pontuação,
    convertendo para minúsculas, e então comparando com "correto" ou "incorreto".
    """
    tokens = output.strip().lower().split()
    if tokens:
        # Remove pontuação do primeiro token
        first_token = tokens[0].strip(string.punctuation)
        if first_token == "correto":
            return "correto"
        elif first_token == "incorreto":
            return "incorreto"
        else:
            return first_token
    return output.strip().lower()

def interactive_test():
    """
    Executa de forma interativa as 5 primeiras perguntas.
    Para cada pergunta, permite que o usuário insira sua resposta livremente,
    avalia-a e exibe se está correta ou incorreta.
    """
    # Lista de 5 perguntas (você pode modificar ou carregar de um arquivo/dataset)
    questions = [
        "Qual a definição legal de 'autoestrada' segundo o Código da Estrada de Portugal?",
        "Qual a velocidade mínima instantânea que devem manter os veículos nas autoestradas portuguesas?",
        "Qual é a regra geral para a posição de marcha segundo o Código da Estrada?",
        "Como é definida 'paragem' pelo Código da Estrada de Portugal?",
        "Qual é a distância lateral mínima que um condutor deve manter ao ultrapassar um velocípede?"
    ]
    
    print("Iniciando teste interativo. Responda às 5 perguntas abaixo:\n")
    
    for idx, question in enumerate(questions, 1):
        print(f"Pergunta {idx}:")
        print(question)
        user_answer = input("Sua resposta: ")
        evaluation = evaluate_answer(question, user_answer)
        normalized_eval = normalize_output(evaluation)
        if normalized_eval == "correto":
            print("Avaliação: Correto!")
        elif normalized_eval == "incorreto":
            print("Avaliação: Incorreto!")
        else:
            print("Avaliação (texto completo):", evaluation)
        print("-" * 80)

if __name__ == '__main__':
    interactive_test()
