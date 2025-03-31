import torch
import json
import random
import os
import re
from collections import Counter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from colorama import Fore, Style, init

# Inicializa o colorama para dar suporte a cores no terminal
init()

class ChatBotCodigoEstrada:
    def __init__(self, model_dir="./trained_model", questions_file="questions_dataset_enhanced.json"):
        print(f"{Fore.YELLOW}Carregando o modelo e recursos...{Style.RESET_ALL}")
        
        # Carrega o modelo e o tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        
        # Configura o dispositivo (GPU se disponível, senão CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{Fore.CYAN}Usando dispositivo: {self.device}{Style.RESET_ALL}")
        self.model.to(self.device)
        
        # Coloca o modelo em modo de avaliação
        self.model.eval()
        
        # Carrega o conjunto de perguntas para o modo quiz
        try:
            with open(questions_file, encoding="utf-8") as f:
                self.questions = json.load(f)
            print(f"{Fore.GREEN}✓ Perguntas para quiz carregadas ({len(self.questions)} questões){Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Erro ao carregar perguntas: {e}{Style.RESET_ALL}")
            self.questions = []
        
        # Carrega o código da estrada para referência rápida
        try:
            with open("Codigo_Estrada_converted.json", encoding="utf-8") as f:
                self.codigo = json.load(f)
            print(f"{Fore.GREEN}✓ Código da Estrada carregado com sucesso{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Erro ao carregar Código da Estrada: {e}{Style.RESET_ALL}")
            self.codigo = {"articles": []}
        
        # Define dicionário de sinônimos para palavras-chave
        self.sinonimos = {
            # Verbos de movimento/ação
            'assinala': ['indica', 'sinaliza', 'mostra', 'marca', 'sinaliza', 'vai mudar', 'pretende mudar', 'está mudando'],
            'mudança': ['alteração', 'troca', 'virar', 'mudar', 'virada', 'conversão', 'direcção', 'direção'],
            'manter': ['conservar', 'guardar', 'preservar', 'sustentar', 'ter'],
            'circular': ['andar', 'trafegar', 'transitar', 'conduzir', 'rodar', 'passar', 'ir', 'fazer-se'],
            'conduzir': ['dirigir', 'guiar', 'pilotar', 'levar'],
            'ultrapassar': ['passar', 'atravessar', 'adiantar', 'exceder'],
            
            # Unidades de medida
            'km/h': ['quilómetros por hora', 'quilometros por hora', 'kms por hora', 'km por hora', 'kmh', 'quilómetro por hora', 'kms/h', 'km p/h', 'quilômetros/hora'],
            
            # Distâncias e quantidades
            '1,5': ['um e meio', 'um metro e meio', '1.5', 'um vírgula cinco', 'um e meio', 'um metro e cinquenta', 'um e cinquenta'],
            
            # Direcções
            'esquerda': ['lado esquerdo', 'à esquerda'],
            'direita': ['lado direito', 'à direita'],
            
            # Veículos
            'veículo': ['carro', 'automóvel', 'viatura'],
            'velocípede': ['bicicleta', 'bike', 'ciclo'],
            
            # Estradas
            'faixa': ['via', 'pista', 'estrada'],
            'rodagem': ['circulação', 'tráfego', 'trânsito'],
        }
            
        print(f"{Fore.GREEN}Modelo carregado e pronto para conversar!{Style.RESET_ALL}")
    
    def gerar_resposta(self, prompt, max_length=150):
        """Gera uma resposta baseada no prompt usando o modelo treinado"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Evitar avisos de gradiente
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                num_return_sequences=1,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodifica a saída e remove o prompt original da resposta
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        resposta = generated_text[len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
        
        # Limpa a resposta para evitar texto incompleto ou excessivo
        if "Pergunta:" in resposta:
            resposta = resposta.split("Pergunta:")[0].strip()
        
        return resposta.strip()
    
    def fazer_pergunta(self):
        """Seleciona uma pergunta aleatória do banco de questões"""
        if not self.questions:
            return None
        
        pergunta = random.choice(self.questions)
        return pergunta
    
    def verificar_resposta(self, pergunta, resposta_usuario):
        """Verifica se a resposta do usuário está correta"""
        resposta_usuario = resposta_usuario.lower().strip()
        
        # Tentativa com método aprimorado
        for resposta_correta in pergunta["respostas_corretas"]:
            if self.similar_avancado(resposta_usuario, resposta_correta.lower()):
                return True, resposta_correta
        
        # Verificar palavras-chave na pergunta e resposta
        pergunta_texto = pergunta["pergunta"].lower()
        if "velocidade" in pergunta_texto and "km" in resposta_usuario:
            # Extrair números da resposta do usuário
            numeros_usuario = re.findall(r'\d+', resposta_usuario)
            for resposta_correta in pergunta["respostas_corretas"]:
                numeros_corretos = re.findall(r'\d+', resposta_correta.lower())
                if numeros_usuario and numeros_corretos and numeros_usuario[0] == numeros_corretos[0]:
                    return True, resposta_correta
                
        # Identifica possíveis respostas incorretas fornecidas para feedback
        for resposta_incorreta in pergunta["respostas_incorretas"]:
            if self.similar_avancado(resposta_usuario, resposta_incorreta.lower()):
                return False, resposta_incorreta
        
        # Se não corresponder a nenhuma das opções
        return False, pergunta["respostas_corretas"][0]
    
    def similar(self, texto1, texto2, threshold=0.7):
        """Método antigo - mantido para referência"""
        palavras1 = set(texto1.split())
        palavras2 = set(texto2.split())
        
        if not palavras1 or not palavras2:
            return False
        
        intersection = palavras1.intersection(palavras2)
        return len(intersection) / max(len(palavras1), len(palavras2)) >= threshold
    
    def normalizar_texto(self, texto):
        """Normaliza o texto para comparação"""
        # Converte para minúsculas e remove espaços extras
        texto = texto.lower().strip()
        
        # Normaliza formatos de velocidade
        texto = re.sub(r'(\d+)\s*(?:km|kms)(?:\s*[-/]?\s*|\s+)(?:p\s*\/?\s*h|por\s+hora|h)', r'\1 km/h', texto)
        
        # Normaliza formatos de distância
        texto = re.sub(r'(\d+)[\.,]5', r'\1,5', texto)
        texto = re.sub(r'um\s+metro\s+e\s+(?:meio|cinquenta)', r'1,5 metros', texto)
        texto = re.sub(r'um\s+e\s+(?:meio|cinquenta)', r'1,5', texto)
        
        # Remove pontuação e caracteres especiais, mas preserva dígitos e vírgulas em números
        texto = re.sub(r'[^\w\s\d,\./áàâãéèêíïóôõöúçñ]', '', texto)
        
        return texto
    
    def expandir_sinonimos(self, palavras):
        """Expande cada palavra para incluir seus sinônimos"""
        expandido = []
        for palavra in palavras:
            expandido.append(palavra)
            # Adiciona sinônimos se existirem
            for chave, sinonimos in self.sinonimos.items():
                if palavra == chave or palavra in sinonimos:
                    expandido.extend([s for s in sinonimos + [chave] if s != palavra])
        return expandido
    
    def similar_improved(self, texto1, texto2, threshold=0.5):
        """Versão melhorada para comparar respostas"""
        # Pré-processamento: normalização
        texto1 = texto1.lower().strip()
        texto2 = texto2.lower().strip()
        
        # Se os textos são muito curtos, faz comparação direta
        if len(texto1) <= 3 or len(texto2) <= 3:
            return texto1 == texto2
        
        # Remove pontuação e caracteres especiais
        texto1 = re.sub(r'[^\w\sáàâãéèêíïóôõöúçñ]', '', texto1)
        texto2 = re.sub(r'[^\w\sáàâãéèêíïóôõöúçñ]', '', texto2)
        
        # Lista de palavras comuns a ignorar
        stop_words = {'o', 'a', 'os', 'as', 'um', 'uma', 'de', 'da', 'do', 'das', 'dos', 'em', 
                      'na', 'no', 'nas', 'nos', 'à', 'ao', 'e', 'é', 'são', 'com', 'para', 'por',
                      'pelo', 'pela', 'seu', 'sua', 'seus', 'suas', 'que', 'se', 'ou'}
        
        # Divide em palavras e remove stop words
        palavras1 = [p for p in texto1.split() if p not in stop_words]
        palavras2 = [p for p in texto2.split() if p not in stop_words]
        
        if not palavras1 or not palavras2:
            return False
        
        # Usando multisets (Counter) para considerar frequência de palavras
        contador1 = Counter(palavras1)
        contador2 = Counter(palavras2)
        
        # Calcula a interseção considerando a frequência
        intersection = sum((contador1 & contador2).values())
        total_words = sum(contador1.values()) + sum(contador2.values())
        
        # Se uma resposta contém a outra quase completamente, considerar similar
        if (intersection / sum(contador1.values()) > 0.8 or 
            intersection / sum(contador2.values()) > 0.8):
            return True
        
        # Calcula a similaridade de Jaccard ponderada
        similarity = intersection / (total_words - intersection) if total_words > intersection else 0
        
        # Debug - opcional, comente para uso normal
        # print(f"Comparando: '{texto1}' e '{texto2}'")
        # print(f"Palavras relevantes: {palavras1} e {palavras2}")
        # print(f"Similaridade: {similarity} (threshold: {threshold})")
        
        return similarity >= threshold
    
    def similar_avancado(self, texto1, texto2, threshold=0.4):
        """Versão avançada para comparar respostas com suporte a sinônimos e normalização de formatos"""
        # Normalização de formatos especiais
        texto1 = self.normalizar_texto(texto1)
        texto2 = self.normalizar_texto(texto2)
        
        # Debug
        # print(f"Normalizado: '{texto1}' e '{texto2}'")
        
        # Se os textos são muito curtos, faz comparação direta
        if len(texto1) <= 3 or len(texto2) <= 3:
            return texto1 == texto2
        
        # Lista de palavras comuns a ignorar
        stop_words = {'o', 'a', 'os', 'as', 'um', 'uma', 'de', 'da', 'do', 'das', 'dos', 'em', 
                      'na', 'no', 'nas', 'nos', 'à', 'ao', 'e', 'é', 'são', 'com', 'para', 'por',
                      'pelo', 'pela', 'seu', 'sua', 'seus', 'suas', 'que', 'se', 'ou', 'quando',
                      'como', 'onde'}
        
        # Divide em palavras e remove stop words
        palavras1 = [p for p in texto1.split() if p not in stop_words]
        palavras2 = [p for p in texto2.split() if p not in stop_words]
        
        if not palavras1 or not palavras2:
            return False
        
        # Expande palavras para incluir sinônimos
        palavras1_exp = self.expandir_sinonimos(palavras1)
        palavras2_exp = self.expandir_sinonimos(palavras2)
        
        # Debug
        # print(f"Palavras expandidas 1: {palavras1_exp}")
        # print(f"Palavras expandidas 2: {palavras2_exp}")
        
        # Conta palavras, incluindo sinônimos
        contador1 = Counter(palavras1_exp)
        contador2 = Counter(palavras2_exp)
        
        # Calcula a interseção considerando a frequência e sinônimos
        intersection = 0
        for p1, count1 in contador1.items():
            for p2, count2 in contador2.items():
                # Verifica se as palavras são iguais ou sinônimas
                if p1 == p2 or self.sao_sinonimos(p1, p2):
                    intersection += min(count1, count2)
                    break
        
        total_words = sum(contador1.values()) + sum(contador2.values())
        
        # Se há muitas palavras em comum, considerar similar
        similarity = (2 * intersection) / total_words if total_words > 0 else 0
        
        # Se uma resposta contém a outra quase completamente, considerar similar
        palavras_unicas1 = set(palavras1)
        palavras_unicas2 = set(palavras2)
        
        # Verifica palavras-chave essenciais
        for p1 in palavras_unicas1:
            if any(self.sao_sinonimos(p1, p2) for p2 in palavras_unicas2):
                similarity += 0.1  # Bônus para cada palavra-chave encontrada
        
        # Debug
        # print(f"Comparando: '{texto1}' e '{texto2}'")
        # print(f"Similaridade: {similarity} (threshold: {threshold})")
        
        # Comparação direta de números (útil para velocidades, distâncias, etc.)
        numeros1 = re.findall(r'\d+', texto1)
        numeros2 = re.findall(r'\d+', texto2)
        if numeros1 and numeros2 and set(numeros1) == set(numeros2):
            similarity += 0.3  # Bônus significativo se os números são iguais
            
        # Verifica padrões específicos
        # Velocidade
        if re.search(r'\b\d+\s*km\/h\b', texto1) and re.search(r'\b\d+\s*km\/h\b', texto2):
            v1 = re.findall(r'\b(\d+)\s*km\/h\b', texto1)
            v2 = re.findall(r'\b(\d+)\s*km\/h\b', texto2)
            if v1 and v2 and v1[0] == v2[0]:
                return True
        
        # Verifica pela direita/esquerda em contextos de ultrapassagem
        if "ultrapassar" in texto1 or "ultrapassar" in texto2:
            if ("direita" in texto1 and "direita" in texto2) or ("esquerda" in texto1 and "esquerda" in texto2):
                similarity += 0.3
                
        # Verifica distância para ciclistas/velocípedes
        if ("velocípede" in texto1 or "bicicleta" in texto1) and ("metro" in texto1 or "1,5" in texto1):
            if ("velocípede" in texto2 or "bicicleta" in texto2) and ("metro" in texto2 or "1,5" in texto2):
                similarity += 0.3
        
        # Se for extremamente similar, aceitar
        return similarity >= threshold
    
    def sao_sinonimos(self, palavra1, palavra2):
        """Verifica se duas palavras são sinônimas"""
        if palavra1 == palavra2:
            return True
            
        for chave, sinonimos in self.sinonimos.items():
            if (palavra1 == chave and palavra2 in sinonimos) or (palavra2 == chave and palavra1 in sinonimos):
                return True
            if palavra1 in sinonimos and palavra2 in sinonimos:
                return True
                
        return False
    
    def buscar_artigo(self, consulta):
        """Busca informações sobre um artigo específico no Código da Estrada"""
        for artigo in self.codigo.get("articles", []):
            if consulta.lower() in artigo.get("article", "").lower():
                return artigo
        return None

    def executar_chat(self):
        """Inicia o chat interativo"""
        print(f"{Fore.CYAN}="*70)
        print(" Bem-vindo ao ChatBot do Código da Estrada Português!")
        print(f" Digite '{Fore.YELLOW}quiz{Style.RESET_ALL}{Fore.CYAN}' para iniciar modo de perguntas e respostas")
        print(f" Digite '{Fore.YELLOW}artigo X{Style.RESET_ALL}{Fore.CYAN}' para consultar um artigo específico")
        print(f" Digite '{Fore.YELLOW}ajuda{Style.RESET_ALL}{Fore.CYAN}' para ver opções disponíveis")
        print(f" Digite '{Fore.YELLOW}sair{Style.RESET_ALL}{Fore.CYAN}' para terminar o chat")
        print("="*70)
        
        modo_quiz = False
        pergunta_atual = None
        
        while True:
            if modo_quiz and not pergunta_atual:
                # Seleciona uma nova pergunta em modo quiz
                pergunta_atual = self.fazer_pergunta()
                if pergunta_atual:
                    print(f"\n{Fore.GREEN}[QUIZ] {pergunta_atual['pergunta']}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}(Digite 'encerrar quiz' para voltar ao modo normal){Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Erro: Não foi possível carregar perguntas para o quiz.{Style.RESET_ALL}")
                    modo_quiz = False
                    continue
            else:
                # Modo normal de chat
                print(f"\n{Fore.CYAN}Você:{Style.RESET_ALL}", end=" ")
            
            try:
                entrada = input().strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{Fore.YELLOW}Chat encerrado. Até logo!{Style.RESET_ALL}")
                break
            
            # Comandos especiais
            if entrada.lower() == "sair":
                print(f"{Fore.YELLOW}Até logo! Conduza com segurança.{Style.RESET_ALL}")
                break
                
            elif entrada.lower() == "ajuda":
                print(f"{Fore.CYAN}Comandos disponíveis:{Style.RESET_ALL}")
                print(f"- {Fore.YELLOW}quiz{Style.RESET_ALL}: Entra no modo de testes com perguntas aleatórias")
                print(f"- {Fore.YELLOW}artigo X{Style.RESET_ALL}: Consulta o Artigo X° do Código da Estrada")
                print(f"- {Fore.YELLOW}encerrar quiz{Style.RESET_ALL}: Sai do modo quiz")
                print(f"- {Fore.YELLOW}ajuda{Style.RESET_ALL}: Mostra esta mensagem")
                print(f"- {Fore.YELLOW}sair{Style.RESET_ALL}: Encerra a conversa")
                continue
                
            elif entrada.lower() == "quiz":
                modo_quiz = True
                pergunta_atual = None
                print(f"{Fore.GREEN}Entrando no modo QUIZ. Vou testar seus conhecimentos!{Style.RESET_ALL}")
                continue
                
            elif entrada.lower() == "encerrar quiz":
                if modo_quiz:
                    modo_quiz = False
                    pergunta_atual = None
                    print(f"{Fore.GREEN}Saindo do modo QUIZ. Voltando ao modo de conversa normal.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Você não está no modo QUIZ atualmente.{Style.RESET_ALL}")
                continue
                
            elif entrada.lower().startswith("artigo"):
                # Buscar informações sobre artigo específico
                try:
                    artigo_num = entrada.split("artigo")[1].strip().lower()
                    artigo = self.buscar_artigo(artigo_num)
                    if artigo:
                        print(f"{Fore.GREEN}Artigo {artigo['article']}{Style.RESET_ALL}")
                        print(f"{Fore.WHITE}{artigo['text']}{Style.RESET_ALL}")
                        print(f"\n{Fore.YELLOW}Referência: {artigo['reference']}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Artigo não encontrado. Tente especificar melhor (ex: 'artigo 1º'){Style.RESET_ALL}")
                except:
                    print(f"{Fore.RED}Formato inválido. Use 'artigo X°' ou 'artigo X'.{Style.RESET_ALL}")
                continue
            
            # Processamento normal ou verificação de resposta no modo quiz
            if modo_quiz and pergunta_atual:
                # Verificar resposta no modo quiz
                if entrada.lower() in ["não sei", "nao sei", "ns"]:
                    print(f"{Fore.YELLOW}Sem problemas! A resposta correta é:{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}{pergunta_atual['respostas_corretas'][0]}{Style.RESET_ALL}")
                    if "referencia" in pergunta_atual:
                        print(f"{Fore.YELLOW}Referência: {pergunta_atual['referencia']}{Style.RESET_ALL}")
                else:
                    correto, resposta_padrao = self.verificar_resposta(pergunta_atual, entrada)
                    
                    if correto:
                        print(f"{Fore.GREEN}✓ Correto! {Style.RESET_ALL}")
                        print(f"{Fore.CYAN}Resposta: {resposta_padrao}{Style.RESET_ALL}")
                        if "referencia" in pergunta_atual:
                            print(f"{Fore.YELLOW}Referência: {pergunta_atual['referencia']}{Style.RESET_ALL}")
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
                        pergunta_atual = None  # Isso fará com que uma nova pergunta seja selecionada
                    else:
                        modo_quiz = False
                        print(f"{Fore.GREEN}Saindo do modo QUIZ. Voltando ao modo de conversa normal.{Style.RESET_ALL}")
                except (KeyboardInterrupt, EOFError):
                    print(f"\n{Fore.YELLOW}Chat encerrado. Até logo!{Style.RESET_ALL}")
                    break
            else:
                # Modo de conversa normal
                prompt = f"Pergunta: {entrada}\nResposta:"
                resposta = self.gerar_resposta(prompt)
                print(f"{Fore.MAGENTA}ChatBot:{Style.RESET_ALL} {resposta}")

if __name__ == "__main__":
    # Verifica se o modelo existe
    model_dir = "./trained_model"
    if not os.path.exists(model_dir):
        print(f"{Fore.RED}Erro: O diretório do modelo '{model_dir}' não foi encontrado.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Certifique-se de que o modelo foi treinado antes de executar o chat.{Style.RESET_ALL}")
        exit(1)
        
    # Inicia o chatbot
    try:
        chatbot = ChatBotCodigoEstrada()
        chatbot.executar_chat()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Chat encerrado pelo usuário. Até logo!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Erro inesperado: {e}{Style.RESET_ALL}") 