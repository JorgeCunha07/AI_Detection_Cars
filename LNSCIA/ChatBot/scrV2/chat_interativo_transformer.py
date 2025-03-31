import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import sys
import os
import random
from colorama import Fore, Style, init

# Inicializa o colorama para cores no terminal
init()

# Verificar disponibilidade de GPU
if not torch.cuda.is_available():
    print("ERRO: GPU não encontrada! Este script requer uma GPU para inferência.")
    print("Por favor, certifique-se de que:")
    print("1. Tem uma GPU NVIDIA instalada")
    print("2. Tem os drivers CUDA instalados")
    print("3. Tem o PyTorch com suporte CUDA instalado")
    sys.exit(1)

# Configurar device para GPU
device = torch.device("cuda")
print(f"Usando GPU: {torch.cuda.get_device_name(0)}")

# Carrega o tokenizer e o modelo
model_name = "microsoft/phi-2"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Configura o token de padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Usa precisão reduzida para melhor performance
        device_map="auto"  # Gerencia automaticamente o carregamento na GPU
    )
    print(f"Modelo {model_name} carregado com sucesso")
except Exception as e:
    print(f"ERRO ao carregar o modelo: {e}")
    sys.exit(1)

def generate_response(prompt, historico_conversa=None, max_length=200, temperature=0.7, top_p=0.9):
    # Prepara o prompt com o histórico da conversa
    if historico_conversa:
        prompt_completo = "\n".join(historico_conversa) + "\nUtilizador: " + prompt
    else:
        prompt_completo = prompt
    
    # Adiciona instruções específicas para português de Portugal
    prompt_completo = """Instruções: Responda sempre em português de Portugal, usando termos e expressões comuns em Portugal. 
Mantenha um tom profissional mas amigável. Use termos específicos de Portugal como 'peço desculpa', 'obrigado', etc.

""" + prompt_completo
    
    # Tokeniza o prompt
    inputs = tokenizer(
        prompt_completo,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Gera a resposta
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        length_penalty=1.0,
        repetition_penalty=1.2,
        min_length=10,
        max_new_tokens=100
    )
    
    # Decodifica a resposta
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove o prompt da resposta
    response = response[len(prompt_completo):].strip()
    
    # Limpa a resposta de possíveis artefatos
    response = response.replace("Pergunta:", "").replace("Resposta:", "").strip()
    
    # Se a resposta estiver vazia ou for muito curta, gera uma resposta padrão
    if not response or len(response.split()) < 3:
        response = "Olá! Sou o seu assistente especializado em condução e Código da Estrada. Como posso ajudar?"
    
    # Remove caracteres especiais e emojis indesejados
    import re
    response = re.sub(r'[^\w\s.,!?áéíóúâêîôûãõàèìòùäëïöüçÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÄËÏÖÜÇ]', '', response)
    
    # Verifica se a resposta está em português de Portugal
    palavras_portuguesas = {
        'olá', 'oi', 'bom', 'dia', 'tarde', 'noite', 'como', 'está', 'tudo', 'bem',
        'obrigado', 'obrigada', 'por favor', 'desculpe', 'sim', 'não', 'pode', 'ajudar',
        'quero', 'saber', 'sobre', 'código', 'estrada', 'condução', 'carro', 'moto',
        'velocidade', 'multa', 'sinal', 'trânsito', 'condutor', 'peão', 'via', 'rua',
        'estrada', 'autoestrada', 'rotunda', 'cruzamento', 'semáforo', 'stop', 'ceda',
        'proibido', 'obrigatório', 'permitido', 'multa', 'coima', 'infração', 'contraordenação'
    }
    
    # Palavras específicas de Portugal
    palavras_portugal = {
        'peço desculpa', 'desculpe', 'obrigado', 'obrigada', 'bom dia', 'boa tarde',
        'boa noite', 'como está', 'tudo bem', 'pode ser', 'claro', 'naturalmente',
        'sem dúvida', 'com certeza', 'exatamente', 'precisamente', 'portanto',
        'consequentemente', 'assim sendo', 'deste modo', 'desta forma'
    }
    
    palavras_resposta = set(response.lower().split())
    
    # Se não houver palavras em português na resposta, gera uma resposta padrão
    if not any(palavra in palavras_portuguesas for palavra in palavras_resposta):
        response = "Olá! Sou o seu assistente especializado em condução e Código da Estrada. Como posso ajudar?"
    
    # Tenta garantir que a resposta use termos de Portugal
    if not any(palavra in palavras_portugal for palavra in palavras_resposta):
        # Adiciona uma frase introdutória em português de Portugal
        response = "Peço desculpa, mas vou reformular a minha resposta. " + response
    
    return response

def load_quiz_data():
    """Carrega as perguntas do quiz do arquivo JSON"""
    try:
        with open("../../Documentacao/Questoes/questoes.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Erro ao carregar perguntas: {e}{Style.RESET_ALL}")
        return []

def load_codigo_data():
    """Carrega o código da estrada do arquivo JSON"""
    try:
        with open("../../Documentacao/BomCondutor/Codigo_Estrada.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Erro ao carregar Código da Estrada: {e}{Style.RESET_ALL}")
        return {"articles": []}

def similar_text(texto1, texto2, threshold=0.6):
    """Compara dois textos e retorna True se forem similares"""
    # Pré-processamento
    texto1 = texto1.lower().strip()
    texto2 = texto2.lower().strip()
    
    # Normaliza formatos de velocidade
    def normalizar_velocidade(texto):
        # Padroniza diferentes formatos de velocidade
        texto = texto.replace('kms', 'km').replace('quilómetros', 'km').replace('quilometros', 'km')
        texto = texto.replace('por hora', '/h').replace('p/h', '/h').replace('por h', '/h')
        texto = texto.replace(' ', '')  # Remove espaços
        return texto
    
    # Normaliza formatos de distância
    def normalizar_distancia(texto):
        # Padroniza diferentes formatos de distância
        texto = texto.replace('metro', 'm').replace('metros', 'm')
        texto = texto.replace('meio', '0.5').replace('e meio', '0.5')
        texto = texto.replace('um', '1').replace('uma', '1')
        texto = texto.replace('dois', '2').replace('duas', '2')
        texto = texto.replace('três', '3').replace('tres', '3')
        texto = texto.replace('quatro', '4')
        texto = texto.replace('cinco', '5')
        texto = texto.replace('seis', '6')
        texto = texto.replace('sete', '7')
        texto = texto.replace('oito', '8')
        texto = texto.replace('nove', '9')
        texto = texto.replace('dez', '10')
        texto = texto.replace(' ', '')  # Remove espaços
        return texto
    
    # Normaliza formatos de direção
    def normalizar_direcao(texto):
        # Padroniza diferentes formatos de direção
        texto = texto.replace('esquerda', 'esq').replace('direita', 'dir')
        texto = texto.replace('à', 'a').replace('á', 'a')
        texto = texto.replace('à frente', 'afrente').replace('a frente', 'afrente')
        texto = texto.replace(' ', '')  # Remove espaços
        return texto
    
    # Aplica todas as normalizações
    texto1 = normalizar_velocidade(texto1)
    texto2 = normalizar_velocidade(texto2)
    texto1 = normalizar_distancia(texto1)
    texto2 = normalizar_distancia(texto2)
    texto1 = normalizar_direcao(texto1)
    texto2 = normalizar_direcao(texto2)
    
    # Se os textos são exatamente iguais após normalização
    if texto1 == texto2:
        return True
    
    # Remove pontuação e caracteres especiais
    import re
    texto1 = re.sub(r'[^\w\s]', '', texto1)
    texto2 = re.sub(r'[^\w\s]', '', texto2)
    
    # Lista de palavras comuns a ignorar
    stop_words = {
        'o', 'a', 'os', 'as', 'um', 'uma', 'de', 'da', 'do', 'das', 'dos',
        'em', 'na', 'no', 'à', 'ao', 'e', 'é', 'que', 'quando', 'em', 'caso',
        'de', 'perigo', 'iminente', 'apenas', 'sinais', 'sonoros', 'localidades'
    }
    
    # Divide em palavras e remove stop words
    palavras1 = [p for p in texto1.split() if p not in stop_words]
    palavras2 = [p for p in texto2.split() if p not in stop_words]
    
    if not palavras1 or not palavras2:
        return False
    
    # Usando multisets (Counter) para considerar frequência de palavras
    from collections import Counter
    contador1 = Counter(palavras1)
    contador2 = Counter(palavras2)
    
    # Calcula a interseção considerando a frequência
    intersection = sum((contador1 & contador2).values())
    
    # Calcula a similaridade de Jaccard ponderada
    similarity = intersection / (sum(contador1.values()) + sum(contador2.values()) - intersection)
    
    # Se a similaridade é alta o suficiente
    if similarity >= threshold:
        return True
    
    # Verifica se há números nos textos
    numeros1 = re.findall(r'\d+\.?\d*', texto1)
    numeros2 = re.findall(r'\d+\.?\d*', texto2)
    
    # Se ambos os textos contêm os mesmos números
    if numeros1 and numeros2:
        # Converte para float para comparar números decimais
        try:
            nums1 = [float(n) for n in numeros1]
            nums2 = [float(n) for n in numeros2]
            if set(nums1) == set(nums2):
                # Se os números são iguais, considera similar mesmo com threshold mais baixo
                return similarity >= (threshold * 0.8)
        except ValueError:
            pass
    
    return False

def main():
    # Carrega dados para quiz e consulta de artigos
    questions = load_quiz_data()
    codigo = load_codigo_data()
    
    print(f"{Fore.CYAN}="*70)
    print(" Bem-vindo ao ChatBot do Código da Estrada Português!")
    print(f" Digite '{Fore.YELLOW}quiz{Style.RESET_ALL}{Fore.CYAN}' para iniciar modo de perguntas e respostas")
    print(f" Digite '{Fore.YELLOW}artigo X{Style.RESET_ALL}{Fore.CYAN}' para consultar um artigo específico")
    print(f" Digite '{Fore.YELLOW}ajuda{Style.RESET_ALL}{Fore.CYAN}' para ver opções disponíveis")
    print(f" Digite '{Fore.YELLOW}sair{Style.RESET_ALL}{Fore.CYAN}' para terminar o chat")
    print(f" Digite '{Fore.YELLOW}conversa{Style.RESET_ALL}{Fore.CYAN}' para iniciar uma conversa normal")
    print("="*70)
    
    modo_quiz = False
    modo_conversa = False
    pergunta_atual = None
    historico_conversa = []
    perguntas_usadas = set()  # Conjunto para rastrear perguntas já usadas
    
    # Adiciona uma mensagem inicial ao histórico
    historico_conversa.append("Assistente: Olá! Sou o assistente do Código da Estrada. Como posso ajudar você hoje?")
    
    while True:
        if modo_quiz and not pergunta_atual:
            # Seleciona uma nova pergunta em modo quiz, evitando repetições
            perguntas_disponiveis = [q for q in questions if q['pergunta'] not in perguntas_usadas]
            if not perguntas_disponiveis:
                print(f"{Fore.YELLOW}Já foram feitas todas as perguntas disponíveis!{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Reiniciando o conjunto de perguntas...{Style.RESET_ALL}")
                perguntas_usadas.clear()
                perguntas_disponiveis = questions
            
            pergunta_atual = random.choice(perguntas_disponiveis)
            perguntas_usadas.add(pergunta_atual['pergunta'])
            
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
            print(f"- {Fore.YELLOW}quiz{Style.RESET_ALL}: Entra no modo de testes com perguntas aleatórias")
            print(f"- {Fore.YELLOW}artigo X{Style.RESET_ALL}: Consulta o Artigo X° do Código da Estrada")
            print(f"- {Fore.YELLOW}conversa{Style.RESET_ALL}: Inicia uma conversa normal")
            print(f"- {Fore.YELLOW}encerrar quiz{Style.RESET_ALL}: Sai do modo quiz")
            print(f"- {Fore.YELLOW}dica{Style.RESET_ALL}: Mostra a resposta correta durante o quiz")
            print(f"- {Fore.YELLOW}ajuda{Style.RESET_ALL}: Mostra esta mensagem")
            print(f"- {Fore.YELLOW}sair{Style.RESET_ALL}: Encerra a conversa")
            continue
            
        elif user_input.lower() == "quiz":
            modo_quiz = True
            modo_conversa = False
            pergunta_atual = None
            historico_conversa = []
            perguntas_usadas.clear()  # Limpa o histórico de perguntas usadas
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
            # Buscar informações sobre artigo específico
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
            except:
                print(f"{Fore.RED}Formato inválido. Use 'artigo X°' ou 'artigo X'.{Style.RESET_ALL}")
            continue
        
        # Processamento normal ou verificação de resposta no modo quiz
        if modo_quiz and pergunta_atual:
            if user_input.lower() == "dica":
                print(f"{Fore.YELLOW}Dica: A resposta correta é:{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{pergunta_atual['respostas_corretas'][0]}{Style.RESET_ALL}")
                if "referencia" in pergunta_atual:
                    print(f"{Fore.YELLOW}Referência: {pergunta_atual['referencia']}{Style.RESET_ALL}")
                continue
            
            # Verificar resposta no modo quiz
            resposta_usuario = user_input.lower().strip()
            correto = False
            
            # Normaliza a resposta do usuário
            resposta_usuario = normalizar_resposta(resposta_usuario)
            
            # Verifica com as respostas corretas
            for resposta_correta in pergunta_atual.get("respostas_corretas", []):
                resposta_correta_norm = normalizar_resposta(resposta_correta.lower())
                if similar_text(resposta_usuario, resposta_correta_norm):
                    correto = True
                    break
            
            if correto:
                print(f"{Fore.GREEN}✓ Correto! {Style.RESET_ALL}")
                print(f"{Fore.CYAN}Resposta: {pergunta_atual['respostas_corretas'][0]}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ Incorreto. {Style.RESET_ALL}")
                print(f"{Fore.CYAN}Resposta correta: {pergunta_atual['respostas_corretas'][0]}{Style.RESET_ALL}")
            
            if "referencia" in pergunta_atual:
                print(f"{Fore.YELLOW}Referência: {pergunta_atual['referencia']}{Style.RESET_ALL}")
            
            # Perguntar se deseja continuar com o quiz
            print(f"\n{Fore.YELLOW}Continuar com o quiz? (s/n){Style.RESET_ALL}")
            try:
                continuar = input().strip().lower()
                if continuar == "s" or continuar == "sim":
                    pergunta_atual = None
                else:
                    modo_quiz = False
                    print(f"{Fore.GREEN}Saindo do modo QUIZ. Voltando ao modo de conversa normal.{Style.RESET_ALL}")
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Fore.YELLOW}Chat encerrado. Até logo!{Style.RESET_ALL}")
                break
        else:
            # Modo de conversa normal
            try:
                # Adiciona a mensagem do usuário ao histórico
                historico_conversa.append(f"Usuário: {user_input}")
                
                # Cria o prompt com o histórico da conversa
                prompt = "\n".join(historico_conversa[-3:]) + "\nAssistente:"
                
                # Gera a resposta
                response = generate_response(prompt, historico_conversa)
                
                # Adiciona a resposta ao histórico
                historico_conversa.append(f"Assistente: {response}")
                
                # Mantém apenas as últimas 6 mensagens (3 pares de pergunta/resposta)
                if len(historico_conversa) > 6:
                    historico_conversa = historico_conversa[-6:]
                
                print(f"{Fore.MAGENTA}ChatBot:{Style.RESET_ALL} {response}")
            except Exception as e:
                print(f"{Fore.RED}Erro ao gerar resposta: {str(e)}{Style.RESET_ALL}")

def normalizar_resposta(texto):
    """Normaliza o texto da resposta para comparação"""
    # Remove acentos e converte para minúsculas
    texto = texto.lower()
    
    # Normaliza espaços
    texto = ' '.join(texto.split())
    
    # Normaliza números e unidades
    texto = texto.replace('€', 'euros').replace('euro', 'euros')
    texto = texto.replace('kms', 'km').replace('km/h', 'km/h').replace('km por hora', 'km/h')
    texto = texto.replace('metros', 'm').replace('metro', 'm')
    texto = texto.replace('1,5', '1.5').replace('1,5', '1.5')
    
    # Remove pontuação
    import re
    texto = re.sub(r'[^\w\s]', '', texto)
    
    return texto.strip()

if __name__ == "__main__":
    main()
    torch.cuda.empty_cache() 