import torch
import json
import random
import os
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
        
        # Compara com as respostas corretas
        for resposta_correta in pergunta["respostas_corretas"]:
            if self.similar(resposta_usuario, resposta_correta.lower()):
                return True, resposta_correta
        
        # Identifica possíveis respostas incorretas fornecidas para feedback
        for resposta_incorreta in pergunta["respostas_incorretas"]:
            if self.similar(resposta_usuario, resposta_incorreta.lower()):
                return False, resposta_incorreta
        
        # Se não corresponder a nenhuma das opções
        return False, pergunta["respostas_corretas"][0]
    
    def similar(self, texto1, texto2, threshold=0.7):
        """Verifica semelhança entre duas strings (implementação simplificada)"""
        # Esta é uma versão básica. Para uma implementação melhor, 
        # considere usar bibliotecas como difflib ou uma métrica de similaridade mais robusta
        palavras1 = set(texto1.split())
        palavras2 = set(texto2.split())
        
        if not palavras1 or not palavras2:
            return False
        
        intersection = palavras1.intersection(palavras2)
        return len(intersection) / max(len(palavras1), len(palavras2)) >= threshold
    
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
            
            entrada = input().strip()
            
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
                correto, resposta_correta = self.verificar_resposta(pergunta_atual, entrada)
                
                if correto:
                    print(f"{Fore.GREEN}✓ Correto! {Style.RESET_ALL}")
                    print(f"{Fore.CYAN}Resposta: {resposta_correta}{Style.RESET_ALL}")
                    if "referencia" in pergunta_atual:
                        print(f"{Fore.YELLOW}Referência: {pergunta_atual['referencia']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}✗ Incorreto. {Style.RESET_ALL}")
                    print(f"{Fore.CYAN}Resposta correta: {resposta_correta}{Style.RESET_ALL}")
                    if "referencia" in pergunta_atual:
                        print(f"{Fore.YELLOW}Referência: {pergunta_atual['referencia']}{Style.RESET_ALL}")
                
                # Perguntar se deseja continuar com o quiz
                print(f"\n{Fore.YELLOW}Continuar com o quiz? (s/n){Style.RESET_ALL}")
                continuar = input().strip().lower()
                if continuar == "s" or continuar == "sim":
                    pergunta_atual = None  # Isso fará com que uma nova pergunta seja selecionada
                else:
                    modo_quiz = False
                    print(f"{Fore.GREEN}Saindo do modo QUIZ. Voltando ao modo de conversa normal.{Style.RESET_ALL}")
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