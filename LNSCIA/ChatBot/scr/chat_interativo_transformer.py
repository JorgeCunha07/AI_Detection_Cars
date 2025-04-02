import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import random
import re
import json  # Import necessário para carregar os JSON
from colorama import Fore, Style, init
from similar_semantic import similar_semantic


# Inicializa o colorama
init(autoreset=True)

# Escolha do dispositivo de inferência
print("Escolha o dispositivo de inferência:")
print("1 - GPU (se disponível)")
print("2 - CPU")
opcao = input("Digite 1 ou 2: ").strip()

if opcao == "1":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"{Fore.GREEN}Usando GPU: {torch.cuda.get_device_name(0)}{Style.RESET_ALL}")
        model_dtype = torch.float16
        device_map = "auto"
    else:
        print(f"{Fore.RED}GPU não encontrada. Utilizando CPU.{Style.RESET_ALL}")
        device = torch.device("cpu")
        model_dtype = torch.float32
        device_map = None
elif opcao == "2":
    device = torch.device("cpu")
    print(f"{Fore.GREEN}Usando CPU para inferência.{Style.RESET_ALL}")
    model_dtype = torch.float32
    device_map = None
else:
    print("Opção inválida. Utilizando CPU por padrão.")
    device = torch.device("cpu")
    model_dtype = torch.float32
    device_map = None

model_path = "./trained_model"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        device_map=device_map
    )
    # Se estiver a usar CPU, garante que o modelo está na CPU
    if device.type == "cpu":
        model.to(device)
    print(f"{Fore.GREEN}Modelo treinado carregado com sucesso de {model_path}{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Erro ao carregar o modelo treinado: {e}{Style.RESET_ALL}")
    print("Tentando carregar o modelo base como fallback...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese")
        model = AutoModelForCausalLM.from_pretrained(
            "pierreguillou/gpt2-small-portuguese",
            torch_dtype=model_dtype,
            device_map=device_map
        )
        if device.type == "cpu":
            model.to(device)
        print(f"{Fore.GREEN}Modelo base carregado com sucesso como fallback{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Erro ao carregar o modelo base: {e}{Style.RESET_ALL}")
        sys.exit(1)

# Verifica e configura o pad_token para garantir que ele não seja igual ao eos_token
if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
print(f"{Fore.YELLOW}Pad token configurado como: {tokenizer.pad_token_id}{Style.RESET_ALL}")

def generate_response(user_input, historico_conversa=None, max_new_tokens=200, temperature=0.7, top_p=0.9):
    # Constrói o prompt completo a partir do histórico
    if historico_conversa:
        prompt_historico = "\n".join(historico_conversa)
    else:
        prompt_historico = f"Usuário: {user_input}"

    # Acrescenta o marcador para o assistente responder
    prompt_completo = prompt_historico + "\nAssistente:"

    # Adiciona as instruções no início
    prompt_completo = """Instruções: Responda sempre em português de Portugal, usando termos e expressões comuns em Portugal.
Mantenha um tom profissional mas amigável. Use termos específicos de Portugal como 'peço desculpa', 'obrigado', etc.

""" + prompt_completo

    print(f"{Fore.CYAN}[DEBUG] Prompt completo:\n{prompt_completo}{Style.RESET_ALL}") # Manter o debug para verificar
    inputs = tokenizer(prompt_completo, return_tensors="pt", add_special_tokens=True, return_attention_mask=True)
    # Substitui eventuais tokens 0 na entrada pelo pad_token_id
    inputs["input_ids"] = inputs["input_ids"].masked_fill(inputs["input_ids"] == 0, tokenizer.pad_token_id)
    inputs = inputs.to(device)
    print(f"[DEBUG] Input IDs: {inputs['input_ids']}")
    print(f"[DEBUG] Attention mask: {inputs['attention_mask']}")

    # Geração ajustada com parâmetros que ajudam a estabilizar os logits
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,         # Utilize sampling para respostas mais naturais
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,       # Evita repetições exageradas
        repetition_penalty=1.0         # Penalidade neutra para repetições
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt_completo):].strip()

    if not response or len(response.split()) < 3:
        response = "Olá! Sou o seu assistente especializado em condução e Código da Estrada. Como posso ajudar?"

    # Limpeza de caracteres indesejados
    response = re.sub(r'[^\w\s.,!?áéíóúâêîôûãõàèìòùäëïöüçÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÜÇ]', '', response)

    # Remover a validação de termos em português
    # palavras_portuguesas = {
    #     'olá', 'oi', 'bom', 'dia', 'tarde', 'noite', 'como', 'está', 'tudo', 'bem',
    #     'obrigado', 'obrigada', 'por favor', 'desculpe', 'sim', 'não', 'pode', 'ajudar',
    #     'quero', 'saber', 'sobre', 'código', 'estrada', 'condução', 'carro', 'moto',
    #     'velocidade', 'multa', 'sinal', 'trânsito', 'condutor', 'peão', 'via', 'rua',
    #     'autoestrada', 'rotunda', 'cruzamento', 'semáforo', 'stop', 'ceda',
    #     'proibido', 'obrigatório', 'permitido', 'coima', 'infração', 'contraordenação'
    # }
    # palavras_portuguesas_extra = {
    #     'peço desculpa', 'desculpe', 'obrigado', 'obrigada', 'bom dia', 'boa tarde',
    #     'boa noite', 'como está', 'tudo bem', 'pode ser', 'claro', 'naturalmente',
    #     'sem dúvida', 'com certeza', 'exatamente', 'precisamente', 'portanto',
    #     'consequentemente', 'assim sendo'
    # }
    # palavras_resposta = set(response.lower().split())
    # if not any(p in palavras_portuguesas for p in palavras_resposta):
    #     response = "Olá! Sou o seu assistente especializado em condução e Código da Estrada. Como posso ajudar?"
    # if not any(p in palavras_portuguesas_extra for p in palavras_resposta):
    #     response = "Peço desculpa, mas vou reformular a minha resposta. " + response

    return response


def load_quiz_data():
    try:
        with open("../../Documentacao/Questoes/questoes.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Erro ao carregar perguntas: {e}{Style.RESET_ALL}")
        return []

def load_codigo_data():
    try:
        with open("../../Documentacao/BomCondutor/Codigo_Estrada.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Erro ao carregar Código da Estrada: {e}{Style.RESET_ALL}")
        return {"articles": []}

def similar_text(texto1, texto2, threshold=0.6):
    texto1 = texto1.lower().strip()
    texto2 = texto2.lower().strip()
    texto1 = re.sub(r'[^\w\s]', '', texto1)
    texto2 = re.sub(r'[^\w\s]', '', texto2)
    stop_words = {'o', 'a', 'os', 'as', 'um', 'uma', 'de', 'da', 'do', 'das', 'dos', 'em', 'na', 'no', 'à', 'ao', 'e', 'é'}
    palavras1 = [p for p in texto1.split() if p not in stop_words]
    palavras2 = [p for p in texto2.split() if p not in stop_words]
    if not palavras1 or not palavras2:
        return False
    from collections import Counter
    contador1 = Counter(palavras1)
    contador2 = Counter(palavras2)
    intersection = sum((contador1 & contador2).values())
    similarity = intersection / (sum(contador1.values()) + sum(contador2.values()) - intersection)
    return similarity >= threshold

def main():
    questions = load_quiz_data()
    codigo = load_codigo_data()
    
    print(f"{Fore.CYAN}{'='*70}")
    print(" Bem-vindo ao ChatBot do Código da Estrada Português!")
    print(f" Digite '{Fore.YELLOW}quiz{Style.RESET_ALL}{Fore.CYAN}' para iniciar modo de perguntas e respostas")
    print(f" Digite '{Fore.YELLOW}artigo X{Style.RESET_ALL}{Fore.CYAN}' para consultar um artigo específico")
    print(f" Digite '{Fore.YELLOW}ajuda{Style.RESET_ALL}{Fore.CYAN}' para ver opções disponíveis")
    print(f" Digite '{Fore.YELLOW}sair{Style.RESET_ALL}{Fore.CYAN}' para terminar o chat")
    print(f" Digite '{Fore.YELLOW}conversa{Style.RESET_ALL}{Fore.CYAN}' para iniciar uma conversa normal")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    modo_quiz = False
    modo_conversa = False
    pergunta_atual = None
    historico_conversa = []
    
    # historico_conversa.append("Assistente: Olá! Sou o assistente do Código da Estrada. Como posso ajudar você hoje?")
    
    while True:
        if modo_quiz and not pergunta_atual:
            pergunta_atual = random.choice(questions) if questions else None
            if pergunta_atual:
                print(f"\n{Fore.GREEN}[QUIZ] {pergunta_atual['pergunta']}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}(Digite 'dica' para ver a resposta correta e continuar respondendo){Style.RESET_ALL}")
                print(f"{Fore.YELLOW}(Digite 'encerrar quiz' para voltar ao modo normal){Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Erro: Não foi possível carregar perguntas para o quiz.{Style.RESET_ALL}")
                modo_quiz = False
                continue
        else:
            print(f"\n{Fore.CYAN}Você:{Style.RESET_ALL}", end=" ")
        
        try:
            user_input = input().strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.YELLOW}Chat encerrado. Até logo!{Style.RESET_ALL}")
            break
        
        if user_input.lower() == "sair":
            print(f"{Fore.YELLOW}Até logo! Conduza com segurança.{Style.RESET_ALL}")
            break
        elif user_input.lower() == "ajuda":
            print(f"{Fore.CYAN}Comandos disponíveis:{Style.RESET_ALL}")
            print(f"- {Fore.YELLOW}quiz{Style.RESET_ALL}: Entra no modo de perguntas e respostas")
            print(f"- {Fore.YELLOW}artigo X{Style.RESET_ALL}: Consulta o artigo X do Código da Estrada")
            print(f"- {Fore.YELLOW}conversa{Style.RESET_ALL}: Inicia uma conversa livre")
            print(f"- {Fore.YELLOW}encerrar quiz{Style.RESET_ALL}: Sai do modo quiz")
            print(f"- {Fore.YELLOW}dica{Style.RESET_ALL}: Mostra a resposta correta durante o quiz")
            print(f"- {Fore.YELLOW}sair{Style.RESET_ALL}: Encerra o chat")
            continue
        elif user_input.lower() == "quiz":
            modo_quiz = True
            modo_conversa = False
            pergunta_atual = None
            historico_conversa = []
            print(f"{Fore.GREEN}Entrando no modo QUIZ. Vou testar seus conhecimentos!{Style.RESET_ALL}")
            continue
        elif user_input.lower() == "conversa":
            modo_quiz = False
            modo_conversa = True
            pergunta_atual = None
            historico_conversa = []
            historico_conversa.append("Assistente: Olá! Podemos conversar sobre qualquer assunto relacionado à condução. Como posso ajudar você hoje?")
            print(f"{Fore.GREEN}Entrando no modo CONVERSA. Podemos conversar sobre qualquer assunto!{Style.RESET_ALL}")
            continue
        elif user_input.lower() == "encerrar quiz":
            if modo_quiz:
                modo_quiz = False
                pergunta_atual = None
                print(f"{Fore.GREEN}Saindo do modo QUIZ. Voltando ao modo de conversa normal.{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Você não está no modo QUIZ atualmente.{Style.RESET_ALL}")
            continue
        elif user_input.lower().startswith("artigo"):
            try:
                artigo_num = user_input.split("artigo")[1].strip().lower()
                for artigo in codigo.get("articles", []):
                    if artigo_num in artigo.get("article", "").lower():
                        print(f"{Fore.GREEN}Artigo {artigo['article']}{Style.RESET_ALL}")
                        print(f"{Fore.WHITE}{artigo['text']}{Style.RESET_ALL}")
                        print(f"\n{Fore.YELLOW}Referência: {artigo['reference']}{Style.RESET_ALL}")
                        break
                else:
                    print(f"{Fore.RED}Artigo não encontrado. Tente especificar melhor (ex: 'artigo 1º'){Style.RESET_ALL}")
            except Exception:
                print(f"{Fore.RED}Formato inválido. Use 'artigo X' (ex: 'artigo 1º'){Style.RESET_ALL}")
            continue
        
        if modo_quiz and pergunta_atual:
            if user_input.lower() == "dica":
                print(f"{Fore.YELLOW}Dica: A resposta correta é:{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{pergunta_atual['respostas_corretas'][0]}{Style.RESET_ALL}")
                if "referencia" in pergunta_atual:
                    print(f"{Fore.YELLOW}Referência: {pergunta_atual['referencia']}{Style.RESET_ALL}")
                continue
            
            resposta_usuario = user_input.lower().strip()
            correto = False
            for resposta_correta in pergunta_atual.get("respostas_corretas", []):
                if similar_semantic(resposta_usuario, resposta_correta.lower()):
                    correto = True
                    break
            if correto:
                print(f"{Fore.GREEN}✓ Correto!{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Resposta: {pergunta_atual['respostas_corretas'][0]}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ Incorreto.{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Resposta correta: {pergunta_atual['respostas_corretas'][0]}{Style.RESET_ALL}")
            if "referencia" in pergunta_atual:
                print(f"{Fore.YELLOW}Referência: {pergunta_atual['referencia']}{Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}Continuar com o quiz? (s/n){Style.RESET_ALL}")
            try:
                continuar = input().strip().lower()
                if continuar in ["s", "sim"]:
                    pergunta_atual = None
                else:
                    modo_quiz = False
                    print(f"{Fore.GREEN}Saindo do modo QUIZ. Voltando ao modo de conversa normal.{Style.RESET_ALL}")
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Fore.YELLOW}Chat encerrado. Até logo!{Style.RESET_ALL}")
                break
        else:
            try:
                historico_conversa.append(f"Usuário: {user_input}")
                # Chama generate_response apenas com a ÚLTIMA entrada do utilizador
                # e o histórico completo. A função generate_response tratará da formatação.
                # Nota: A função generate_response precisa ser ligeiramente ajustada também.
                response = generate_response(user_input, historico_conversa) # Passa user_input diretamente
                historico_conversa.append(f"Assistente: {response}")
                if len(historico_conversa) > 6:
                    historico_conversa = historico_conversa[-6:] # Mantém o limite do histórico
                print(f"{Fore.MAGENTA}ChatBot:{Style.RESET_ALL} {response}")
            except Exception as e:
                print(f"{Fore.RED}Erro ao gerar resposta: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
