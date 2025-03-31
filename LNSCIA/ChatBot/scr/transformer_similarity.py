import torch
import re
import os
from colorama import Fore, Style
from sentence_transformers import SentenceTransformer

class TransformerSimilarity:
    """
    Classe para comparação de similaridade semântica usando modelos transformer
    Especialmente adaptada para o contexto do Código da Estrada Português
    """
    
    def __init__(self, model_name="distiluse-base-multilingual-cased-v1", threshold=0.75):
        """
        Inicializa o modelo transformer para comparação de similaridade semântica
        
        Args:
            model_name (str): Nome do modelo pre-treinado da biblioteca sentence-transformers
                              Opções recomendadas para português:
                                - "distiluse-base-multilingual-cased-v1" (mais rápido, bom em pt-PT)
                                - "paraphrase-multilingual-mpnet-base-v2" (melhor qualidade, mais lento)
                                - "LaBSE" (suporte a 109 línguas, incluindo português)
            threshold (float): Limiar de similaridade para considerar respostas equivalentes (0.0 a 1.0)
        """
        self.modelo_carregado = False
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            print(f"{Fore.YELLOW}Carregando modelo transformer '{model_name}'...{Style.RESET_ALL}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.modelo_carregado = True
            print(f"{Fore.GREEN}✓ Modelo de similaridade carregado com sucesso{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Erro ao carregar modelo: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Certifique-se de que as bibliotecas necessárias estão instaladas:{Style.RESET_ALL}")
            print("pip install sentence-transformers transformers torch numpy")
    
    def normalizar_texto(self, texto):
        """
        Normaliza o texto para melhorar a comparação
        
        Args:
            texto (str): Texto a ser normalizado
            
        Returns:
            str: Texto normalizado
        """
        if not texto:
            return ""
            
        # Converte para minúsculas e remove espaços extras
        texto = texto.lower().strip()
        
        # Normaliza formatos de velocidade para análise posterior
        texto = re.sub(r'(\d+)\s*(?:km|kms)(?:\s*[-/]?\s*|\s+)(?:p\s*\/?\s*h|por\s+hora|h)', r'\1 km/h', texto)
        
        # Normaliza formatos de distância - VERSÃO MELHORADA
        texto = re.sub(r'(\d+)[\.,]5', r'\1,5', texto)
        # Captura tanto "um metro e meio" quanto "1 metro e meio"
        texto = re.sub(r'(?:um|1)\s+metro[s]?\s+e\s+(?:meio|cinquenta|50)', r'1,5 metros', texto)
        texto = re.sub(r'(?:um|1)\s+e\s+(?:meio|cinquenta|50)', r'1,5', texto)
        texto = re.sub(r'metro[s]?\s+e\s+(?:meio|cinquenta|50)', r'1,5 metros', texto)
        
        # Normaliza palavras para ciclista/velocípede
        texto = re.sub(r'\b(?:bicicleta|bike|ciclo)\b', 'velocípede', texto)
        
        return texto
        
    def similar(self, frase1, frase2, threshold=None):
        """
        Verifica se duas frases são semanticamente similares usando embeddings e similaridade coseno
        
        Args:
            frase1 (str): Primeira frase
            frase2 (str): Segunda frase
            threshold (float, optional): Limiar de similaridade. Se None, usa o valor padrão da classe
            
        Returns:
            bool: True se as frases forem similares, False caso contrário
        """
        if not self.modelo_carregado:
            return False
            
        if not threshold:
            threshold = self.threshold
            
        # Normaliza os textos
        frase1_norm = self.normalizar_texto(frase1)
        frase2_norm = self.normalizar_texto(frase2)
        
        # Casos especiais para melhor precisão - VERSÃO MELHORADA
        
        # Verificação de padrões exatos para casos problemáticos
        
        # Para "1,5 metros" e variações como "1 metro e meio"
        padrao_metro_e_meio1 = re.search(r'1,5\s*metros?', frase1_norm) or re.search(r'(?:um|1)\s+metro\s+e\s+meio', frase1)
        padrao_metro_e_meio2 = re.search(r'1,5\s*metros?', frase2_norm) or re.search(r'(?:um|1)\s+metro\s+e\s+meio', frase2)
        
        if padrao_metro_e_meio1 and padrao_metro_e_meio2:
            return True
        
        # Verificação de velocidade (km/h)
        velocidade1 = re.findall(r'\b(\d+)\s*km\/h\b', frase1_norm)
        velocidade2 = re.findall(r'\b(\d+)\s*km\/h\b', frase2_norm)
        if velocidade1 and velocidade2:
            # Se as velocidades são iguais, aumenta o threshold de similaridade
            if velocidade1[0] == velocidade2[0]:
                threshold = max(0.65, threshold - 0.1)  # Torna threshold mais flexível
            else:
                # Velocidades diferentes são respostas diferentes, independente do resto
                return False
        
        # Verificação para distâncias específicas (1,5 metros)
        distancia1 = re.findall(r'\b1,5\s*metros?\b', frase1_norm)
        distancia2 = re.findall(r'\b1,5\s*metros?\b', frase2_norm)
        if (distancia1 and not distancia2) or (not distancia1 and distancia2):
            # Verifica novamente por padrões específicos de distância
            metro_meio1 = re.search(r'metro\s+e\s+meio', frase1.lower()) or re.search(r'1\s*[,\.]\s*5', frase1)
            metro_meio2 = re.search(r'metro\s+e\s+meio', frase2.lower()) or re.search(r'1\s*[,\.]\s*5', frase2)
            
            if (metro_meio1 and distancia2) or (metro_meio2 and distancia1):
                return True
        
        # Verificação de ultrapassagem pela direita/esquerda
        direcao1 = "esquerda" in frase1_norm
        direcao2 = "esquerda" in frase2_norm
        direcao3 = "direita" in frase1_norm
        direcao4 = "direita" in frase2_norm
        
        if (direcao1 and direcao4) or (direcao2 and direcao3):
            # Direções opostas na ultrapassagem são respostas diferentes
            return False
        
        # Gera embeddings usando o modelo
        embedding1 = self.model.encode(frase1_norm, convert_to_tensor=True)
        embedding2 = self.model.encode(frase2_norm, convert_to_tensor=True)
        
        # Calcula similaridade coseno
        cosine_similarity = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), 
                                                                 embedding2.unsqueeze(0))
        similarity_score = cosine_similarity.item()
        
        # Ajuste dinâmico para casos específicos
        # Se ambas as frases mencionam a mesma velocidade, somos mais lenientes
        if velocidade1 and velocidade2 and velocidade1[0] == velocidade2[0]:
            threshold -= 0.05
            
        # Se ambas as frases mencionam a mesma direção, somos mais lenientes
        if (direcao1 and direcao2) or (direcao3 and direcao4):
            threshold -= 0.05
            
        # Imprime informações de debug para desenvolvimento (opcional)
        # print(f"Comparando: '{frase1}' vs '{frase2}'")
        # print(f"Normalizados: '{frase1_norm}' vs '{frase2_norm}'")
        # print(f"Similaridade: {similarity_score:.3f}, Threshold: {threshold:.3f}")
        
        return similarity_score >= threshold
    
    def verificar_resposta(self, resposta_usuario, respostas_corretas, threshold=None):
        """
        Verifica se a resposta do usuário corresponde a alguma das respostas corretas
        
        Args:
            resposta_usuario (str): Resposta fornecida pelo usuário
            respostas_corretas (list): Lista de possíveis respostas corretas
            threshold (float, optional): Limiar de similaridade
            
        Returns:
            tuple: (bool, str) - (é_correta, resposta_original)
        """
        if not self.modelo_carregado:
            return False, respostas_corretas[0] if respostas_corretas else ""
            
        resposta_usuario = resposta_usuario.lower().strip()
        
        # Tenta cada resposta correta
        for resposta_correta in respostas_corretas:
            if self.similar(resposta_usuario, resposta_correta.lower(), threshold):
                return True, resposta_correta
                
        # Se não encontrou nenhuma correspondência
        return False, respostas_corretas[0] if respostas_corretas else ""


# Exemplo de uso
if __name__ == "__main__":
    comparador = TransformerSimilarity()
    
    # Exemplos de perguntas e respostas do Código da Estrada
    perguntas = [
        {
            "pergunta": "Qual é a velocidade máxima permitida para automóveis ligeiros em autoestradas?",
            "respostas_corretas": ["120 km/h"],
            "respostas_usuario": [
                "120 km/h", 
                "120 quilómetros por hora", 
                "cento e vinte quilómetros por hora",
                "É de 120 km por hora",
                "A velocidade máxima é 120"
            ]
        },
        {
            "pergunta": "Qual a distância lateral mínima ao ultrapassar velocípedes?",
            "respostas_corretas": ["1,5 metros"],
            "respostas_usuario": [
                "1,5 metros",
                "Um metro e meio",
                "1,5m",
                "1 metro e meio",  # Adicionado para testar o caso problemático
                "Um metro e cinquenta centímetros",
                "A distância mínima é de 1,5 metros",
                "metro e meio"  # Outra variação comum
            ]
        },
        {
            "pergunta": "Em qual lado devemos circular numa via de sentido único?",
            "respostas_corretas": ["Lado direito"],
            "respostas_usuario": [
                "No lado direito",
                "Devemos circular pelo lado direito",
                "À direita",
                "Pela direita da via"
            ]
        }
    ]
    
    # Testar as respostas
    for pergunta in perguntas:
        print(f"\n{Fore.GREEN}Pergunta: {pergunta['pergunta']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Resposta correta: {pergunta['respostas_corretas'][0]}{Style.RESET_ALL}")
        
        for resposta in pergunta['respostas_usuario']:
            resultado, _ = comparador.verificar_resposta(resposta, pergunta['respostas_corretas'])
            print(f"Resposta: '{resposta}' - {Fore.GREEN if resultado else Fore.RED}{'✓ Correta' if resultado else '✗ Incorreta'}{Style.RESET_ALL}")
        
        # Testar uma resposta incorreta
        resposta_incorreta = "Não sei" if "velocidade" in pergunta['pergunta'] else "2 metros" if "distância" in pergunta['pergunta'] else "lado esquerdo"
        resultado, _ = comparador.verificar_resposta(resposta_incorreta, pergunta['respostas_corretas'])
        print(f"Resposta: '{resposta_incorreta}' - {Fore.GREEN if resultado else Fore.RED}{'✓ Correta' if resultado else '✗ Incorreta (esperado)'}{Style.RESET_ALL}") 