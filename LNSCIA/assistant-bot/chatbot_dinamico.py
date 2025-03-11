import random
import re
import unicodedata
import json
import os
from datetime import datetime

class ChatbotDinamico:
    def __init__(self):
        """Inicializa o chatbot dinâmico."""
        # Carregar ou criar base de conhecimento
        self.conhecimento = self.carregar_conhecimento()
        
        # Saudações possíveis
        self.saudacoes = [
            "Olá! Como posso ajudar?",
            "Oi! Em que posso ser útil hoje?",
            "Olá, tudo bem? Como posso te ajudar?",
            "Oi! Estou aqui para conversar. O que você gostaria de saber?"
        ]
        
        # Respostas para quando não entender
        self.respostas_fallback = [
            "Desculpe, não entendi. Pode explicar de outra forma?",
            "Não tenho certeza do que você quer dizer. Pode reformular?",
            "Hmm, não compreendi bem. Pode dizer de outro jeito?",
            "Ainda estou aprendendo. Pode explicar melhor?"
        ]
        
        # Respostas de despedida
        self.despedidas = [
            "Até logo! Foi um prazer conversar com você.",
            "Adeus! Volte sempre que quiser conversar.",
            "Tchau! Espero ter sido útil.",
            "Até a próxima! Tenha um ótimo dia."
        ]
        
        # Histórico da conversa atual
        self.historico = []
    
    def carregar_conhecimento(self):
        """Carrega a base de conhecimento ou cria uma nova se não existir."""
        arquivo_conhecimento = "conhecimento_chatbot.json"
        
        if os.path.exists(arquivo_conhecimento):
            try:
                with open(arquivo_conhecimento, 'r', encoding='utf-8') as arquivo:
                    return json.load(arquivo)
            except:
                print("Erro ao carregar conhecimento. Criando nova base...")
        
        # Base de conhecimento inicial
        return {
            "perguntas_respostas": [
                {
                    "padrao": r"(?i).*(?:como|tudo bem|como vai|como está).*(?:você|tu|voce).*",
                    "respostas": [
                        "Estou bem, obrigado por perguntar! E você?",
                        "Tudo ótimo! Como você está?",
                        "Estou funcionando perfeitamente! E você?"
                    ]
                },
                {
                    "padrao": r"(?i).*(?:quem|o que).*(?:é|és|e).*(?:você|tu|voce).*",
                    "respostas": [
                        "Sou um chatbot simples criado para conversar com você em português.",
                        "Sou um assistente virtual básico, projetado para dialogar em português.",
                        "Sou um programa de computador feito para simular uma conversa natural."
                    ]
                },
                {
                    "padrao": r"(?i).*(?:obrigad|valeu|thanks).*",
                    "respostas": [
                        "De nada! Estou aqui para ajudar.",
                        "Por nada! Foi um prazer.",
                        "Disponha! Se precisar de mais alguma coisa, é só dizer."
                    ]
                },
                {
                    "padrao": r"(?i).*(?:tempo|clima|chuva).*(?:hoje|amanhã|agora).*",
                    "respostas": [
                        "Desculpe, não tenho acesso a informações de tempo real sobre o clima.",
                        "Não posso verificar o clima, pois não tenho acesso à internet.",
                        "Como sou um programa simples, não consigo verificar previsões do tempo."
                    ]
                },
                {
                    "padrao": r"(?i).*(?:gosta|gosto).*(?:de|do|da).*",
                    "respostas": [
                        "Interessante saber sobre seus gostos! Como chatbot, não tenho preferências reais, mas gosto de aprender com nossas conversas.",
                        "Legal! Embora eu não tenha gostos como humanos, adoro conversar sobre diversos assuntos.",
                        "Que bom! Eu 'gosto' de ajudar pessoas e ter conversas interessantes."
                    ]
                },
                {
                    "padrao": r"(?i)^(oi|olá|ola|hey|e ai|eai|bom dia|boa tarde|boa noite)[\s!]*$",
                    "respostas": [
                        "Olá! Como posso ajudar você hoje?",
                        "Oi! Tudo bem? Em que posso ser útil?",
                        "Olá! É um prazer conversar com você. Como vai?",
                        "Oi! Como está seu dia hoje?"
                    ]
                },
                {
                    "padrao": r"(?i)^(tudo bem|como vai|como está|tudo bom|beleza)[\s?!]*$",
                    "respostas": [
                        "Estou bem, obrigado! E você?",
                        "Tudo ótimo por aqui! E com você?",
                        "Estou funcionando perfeitamente! Como você está?",
                        "Tudo tranquilo! E você, como está?"
                    ]
                },
                {
                    "padrao": r"(?i).*(seu nome|te chama|chama como|como se chama).*",
                    "respostas": [
                        "Eu sou um chatbot dinâmico. Pode me chamar de Bot!",
                        "Meu nome é Bot, sou um assistente virtual em português.",
                        "Sou o Bot, um chatbot feito para conversar em português."
                    ]
                },
                {
                    "padrao": r"(?i).*(horas são|que horas|hora atual|hora agora).*",
                    "respostas": [
                        "Desculpe, não tenho acesso ao relógio do sistema.",
                        "Como sou um programa simples, não consigo verificar a hora atual.",
                        "Não tenho essa funcionalidade implementada ainda."
                    ]
                },
                {
                    "padrao": r"(?i).*(ajuda|ajudar|pode fazer|consegue fazer).*",
                    "respostas": [
                        "Posso conversar sobre diversos assuntos! Sou um chatbot simples, mas estou sempre aprendendo.",
                        "Estou aqui para conversar e aprender com você. Se eu não souber responder algo, você pode me ensinar!",
                        "Posso manter uma conversa e aprender com nossas interações. O que você gostaria de saber?"
                    ]
                }
            ],
            "aprendizado": []
        }
    
    def salvar_conhecimento(self):
        """Salva a base de conhecimento atualizada."""
        with open("conhecimento_chatbot.json", 'w', encoding='utf-8') as arquivo:
            json.dump(self.conhecimento, arquivo, ensure_ascii=False, indent=4)
    
    def normalizar_texto(self, texto):
        """Normaliza o texto removendo acentos e convertendo para minúsculas."""
        # Converter para minúsculas
        texto = texto.lower()
        
        # Remover acentos
        texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                        if not unicodedata.combining(c))
        
        return texto
    
    def extrair_palavras_chave(self, texto):
        """Extrai palavras-chave de um texto, removendo stopwords."""
        # Lista de stopwords em português
        stopwords = [
            "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até",
            "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois",
            "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram",
            "éramos", "essa", "essas", "esse", "esses", "esta", "estas", "este", "estes",
            "eu", "foi", "fomos", "for", "foram", "fosse", "fossem", "fui", "há", "isso",
            "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha",
            "minhas", "muito", "na", "não", "nas", "nem", "no", "nos", "nós", "nossa", "nossas",
            "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo",
            "pelos", "por", "qual", "quando", "que", "quem", "são", "se", "seja", "sejam",
            "sejamos", "sem", "será", "serão", "serei", "seremos", "seria", "seriam", "seríamos",
            "seu", "seus", "só", "somos", "sou", "sua", "suas", "também", "te", "tem", "temos",
            "tenho", "teu", "teus", "tu", "tua", "tuas", "um", "uma", "você", "vocês", "vos"
        ]
        
        # Normalizar texto
        texto_normalizado = self.normalizar_texto(texto)
        
        # Dividir em palavras e remover stopwords
        palavras = [palavra for palavra in texto_normalizado.split() if palavra not in stopwords and len(palavra) > 2]
        
        return palavras
    
    def encontrar_resposta(self, mensagem):
        """Encontra uma resposta adequada para a mensagem do usuário."""
        # Verificar padrões conhecidos
        for item in self.conhecimento["perguntas_respostas"]:
            if re.match(item["padrao"], mensagem):
                return random.choice(item["respostas"])
        
        # Extrair palavras-chave da mensagem
        palavras_chave = self.extrair_palavras_chave(mensagem)
        
        # Verificar aprendizado anterior
        melhores_correspondencias = []
        
        for item in self.conhecimento["aprendizado"]:
            # Extrair palavras-chave da pergunta armazenada
            palavras_chave_armazenadas = self.extrair_palavras_chave(item["pergunta"])
            
            # Calcular correspondência (número de palavras-chave em comum)
            palavras_comuns = set(palavras_chave).intersection(set(palavras_chave_armazenadas))
            
            # Se houver pelo menos 1 palavra-chave em comum
            if len(palavras_comuns) > 0:
                # Calcular pontuação de correspondência (proporção de palavras em comum)
                pontuacao = len(palavras_comuns) / max(len(palavras_chave), len(palavras_chave_armazenadas))
                
                melhores_correspondencias.append({
                    "resposta": item["resposta"],
                    "pontuacao": pontuacao,
                    "data": item["data"]
                })
        
        # Ordenar por pontuação (maior primeiro)
        melhores_correspondencias.sort(key=lambda x: x["pontuacao"], reverse=True)
        
        # Se encontrou correspondências com pontuação mínima
        if melhores_correspondencias and melhores_correspondencias[0]["pontuacao"] >= 0.3:
            return melhores_correspondencias[0]["resposta"]
        
        # Se não encontrar resposta
        return random.choice(self.respostas_fallback)
    
    def aprender(self, pergunta, resposta_usuario):
        """Aprende com a interação do usuário."""
        if len(pergunta.strip()) > 3 and len(resposta_usuario.strip()) > 3:
            # Verificar se já existe uma pergunta muito similar
            palavras_chave_pergunta = self.extrair_palavras_chave(pergunta)
            
            for item in self.conhecimento["aprendizado"]:
                palavras_chave_armazenadas = self.extrair_palavras_chave(item["pergunta"])
                palavras_comuns = set(palavras_chave_pergunta).intersection(set(palavras_chave_armazenadas))
                
                # Se a pergunta for muito similar a uma existente, atualizar a resposta
                if len(palavras_comuns) / max(len(palavras_chave_pergunta), len(palavras_chave_armazenadas)) >= 0.8:
                    item["resposta"] = resposta_usuario
                    item["data"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.salvar_conhecimento()
                    return True
            
            # Se não encontrou pergunta similar, adicionar nova
            self.conhecimento["aprendizado"].append({
                "pergunta": pergunta,
                "resposta": resposta_usuario,
                "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            self.salvar_conhecimento()
            return True
        return False
    
    def processar_mensagem(self, mensagem):
        """Processa a mensagem do usuário e retorna uma resposta."""
        # Verificar se é uma despedida
        if re.match(r"(?i).*(?:adeus|tchau|até logo|ate logo|xau).*", mensagem):
            return random.choice(self.despedidas), True
        
        # Adicionar à história
        self.historico.append({"usuario": mensagem})
        
        # Encontrar resposta
        resposta = self.encontrar_resposta(mensagem)
        
        # Adicionar resposta ao histórico
        self.historico.append({"bot": resposta})
        
        return resposta, False
    
    def iniciar_conversa(self):
        """Inicia a conversa com o usuário."""
        print(random.choice(self.saudacoes))
        
        encerrar = False
        ultima_pergunta = ""
        
        while not encerrar:
            try:
                # Obter mensagem do usuário
                mensagem = input("\nVocê: ").strip()
                
                # Verificar se a mensagem está vazia
                if not mensagem:
                    print("\nBot: Parece que você não digitou nada. Pode tentar novamente?")
                    continue
                
                # Processar mensagem
                resposta, encerrar = self.processar_mensagem(mensagem)
                
                # Exibir resposta
                print(f"\nBot: {resposta}")
                
                # Se o bot não entendeu, perguntar o que deveria responder
                if resposta in self.respostas_fallback and not encerrar:
                    aprender = input("\nGostaria de me ensinar como responder? (s/n): ").strip().lower()
                    if aprender == 's':
                        nova_resposta = input("Como eu deveria ter respondido? ").strip()
                        if self.aprender(mensagem, nova_resposta):
                            print("\nBot: Obrigado! Aprendi algo novo.")
                
                ultima_pergunta = mensagem
            
            except KeyboardInterrupt:
                print("\n\nBot: Parece que você quer encerrar a conversa. Até logo!")
                encerrar = True
            except Exception as e:
                print(f"\nBot: Ops, ocorreu um erro: {str(e)}. Vamos continuar nossa conversa.")

# Executar o chatbot
if __name__ == "__main__":
    print("=== Chatbot Dinâmico em Português ===")
    print("(Digite 'adeus' para encerrar a conversa)")
    print("-------------------------------------")
    
    try:
        chatbot = ChatbotDinamico()
        chatbot.iniciar_conversa()
    except Exception as e:
        print(f"\nErro ao iniciar o chatbot: {str(e)}")
        print("Tente novamente mais tarde.") 