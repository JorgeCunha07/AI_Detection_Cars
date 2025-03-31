#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para comparar os sistemas de similaridade:
1. Baseado em regras (método original)
2. Baseado em transformers (método novo)

Este script utiliza uma série de exemplos do Código da Estrada
e compara os resultados de ambos os sistemas para mostrar as melhorias.
"""

import sys
import json
import time
import os
from colorama import Fore, Style, init

# Inicializa o colorama para cores no terminal
init()

# Importa os sistemas de similaridade
try:
    # Importa o sistema baseado em transformer
    from transformer_similarity import TransformerSimilarity
    transformer_disponivel = True
except ImportError:
    print(f"{Fore.YELLOW}Aviso: Não foi possível importar o sistema baseado em transformer.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Certifique-se de que as dependências estão instaladas.{Style.RESET_ALL}")
    transformer_disponivel = False

# Importa o sistema baseado em regras
class SistemaBaseadoEmRegras:
    def __init__(self):
        # Definição de sinônimos para o sistema baseado em regras
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
    
    def normalizar_texto(self, texto):
        """Normaliza o texto para comparação"""
        import re
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
    
    def similar(self, texto1, texto2, threshold=0.4):
        """Sistema baseado em regras para comparar similaridade"""
        import re
        from collections import Counter
        
        # Normalização de formatos especiais
        texto1 = self.normalizar_texto(texto1)
        texto2 = self.normalizar_texto(texto2)
        
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

# Casos de teste para comparação
CASOS_TESTE = [
    {
        "categoria": "Velocidade",
        "pergunta": "Qual é a velocidade máxima para veículos ligeiros em autoestradas?",
        "resposta_padrao": "120 km/h",
        "variacoes": [
            "120 quilómetros por hora",
            "Cento e vinte quilómetros por hora",
            "A velocidade máxima permitida é de 120 km/h",
            "120kms por hora",
            "120 kms/h",
            "120"
        ],
        "incorretas": [
            "100 km/h",
            "90 quilómetros por hora",
            "Não há limite de velocidade",
            "50 km/h",
            "É permitido 140 km/h"
        ]
    },
    {
        "categoria": "Distância",
        "pergunta": "Qual a distância lateral mínima ao ultrapassar velocípedes?",
        "resposta_padrao": "1,5 metros",
        "variacoes": [
            "Um metro e meio",
            "1,5 m",
            "Um metro e cinquenta centímetros",
            "Deve guardar uma distância lateral de 1,5 metros",
            "A distância mínima lateral é de 1,5 metros"
        ],
        "incorretas": [
            "1 metro",
            "2 metros",
            "50 centímetros",
            "Não há distância mínima definida"
        ]
    },
    {
        "categoria": "Ultrapassagem",
        "pergunta": "Em que casos é permitido ultrapassar pela direita?",
        "resposta_padrao": "Quando o veículo à frente indica mudança de direção para a esquerda",
        "variacoes": [
            "Quando o carro da frente sinaliza que vai virar à esquerda",
            "Se o condutor à nossa frente assinala mudança para o lado esquerdo",
            "Quando o veículo que está à nossa frente indica que vai mudar para a esquerda",
            "Quando o condutor da frente vai virar para a esquerda"
        ],
        "incorretas": [
            "É sempre permitido ultrapassar pela direita",
            "Quando o veículo da frente sinaliza mudança para a direita",
            "Nunca é permitido ultrapassar pela direita",
            "Apenas nas autoestradas"
        ]
    },
    {
        "categoria": "Estacionamento",
        "pergunta": "A que distância mínima das passadeiras é proibido parar ou estacionar?",
        "resposta_padrao": "5 metros",
        "variacoes": [
            "Cinco metros",
            "A 5 metros de distância",
            "É proibido parar ou estacionar a menos de 5 metros das passadeiras",
            "5 m de uma passagem de peões",
            "Uma distância de 5 metros"
        ],
        "incorretas": [
            "3 metros",
            "10 metros",
            "Não é proibido estacionar junto às passadeiras",
            "2 metros é suficiente"
        ]
    }
]

def comparar_sistemas(casos_teste):
    """Compara os resultados dos dois sistemas de similaridade"""
    # Inicializa os sistemas
    sistema_regras = SistemaBaseadoEmRegras()
    
    if transformer_disponivel:
        sistema_transformer = TransformerSimilarity()
    else:
        print(f"{Fore.RED}Sistema Transformer não disponível. Comparação limitada.{Style.RESET_ALL}")
        return
    
    # Resultados globais
    total_corretas_regras = 0
    total_corretas_transformer = 0
    total_variacoes = 0
    total_incorretas_regras = 0
    total_incorretas_transformer = 0
    total_incorretas = 0
    
    print(f"{Fore.CYAN}{'='*100}")
    print(f" COMPARAÇÃO ENTRE SISTEMAS DE SIMILARIDADE")
    print(f"{'='*100}{Style.RESET_ALL}")
    
    # Para cada caso de teste
    for caso in casos_teste:
        print(f"\n{Fore.GREEN}CATEGORIA: {caso['categoria'].upper()}{Style.RESET_ALL}")
        print(f"Pergunta: {caso['pergunta']}")
        print(f"Resposta padrão: {caso['resposta_padrao']}")
        
        # Compara as variações corretas
        print(f"\n{Fore.CYAN}Testando variações corretas:{Style.RESET_ALL}")
        corretas_regras = 0
        corretas_transformer = 0
        
        for i, variacao in enumerate(caso['variacoes'], 1):
            # Mede o tempo para cada sistema
            inicio = time.time()
            resultado_regras = sistema_regras.similar(caso['resposta_padrao'], variacao)
            tempo_regras = time.time() - inicio
            
            inicio = time.time()
            resultado_transformer = sistema_transformer.similar(caso['resposta_padrao'], variacao)
            tempo_transformer = time.time() - inicio
            
            corretas_regras += 1 if resultado_regras else 0
            corretas_transformer += 1 if resultado_transformer else 0
            
            print(f"{i}. \"{variacao}\"")
            print(f"   Sistema baseado em regras: {Fore.GREEN if resultado_regras else Fore.RED}{'✓' if resultado_regras else '✗'}{Style.RESET_ALL} ({tempo_regras:.3f}s)")
            print(f"   Sistema Transformer: {Fore.GREEN if resultado_transformer else Fore.RED}{'✓' if resultado_transformer else '✗'}{Style.RESET_ALL} ({tempo_transformer:.3f}s)")
        
        # Compara as variações incorretas
        print(f"\n{Fore.CYAN}Testando respostas incorretas:{Style.RESET_ALL}")
        falsos_positivos_regras = 0
        falsos_positivos_transformer = 0
        
        for i, incorreta in enumerate(caso['incorretas'], 1):
            # Mede o tempo para cada sistema
            inicio = time.time()
            resultado_regras = sistema_regras.similar(caso['resposta_padrao'], incorreta)
            tempo_regras = time.time() - inicio
            
            inicio = time.time()
            resultado_transformer = sistema_transformer.similar(caso['resposta_padrao'], incorreta)
            tempo_transformer = time.time() - inicio
            
            falsos_positivos_regras += 1 if resultado_regras else 0
            falsos_positivos_transformer += 1 if resultado_transformer else 0
            
            print(f"{i}. \"{incorreta}\"")
            print(f"   Sistema baseado em regras: {Fore.RED if resultado_regras else Fore.GREEN}{'✓' if not resultado_regras else '✗'}{Style.RESET_ALL} ({tempo_regras:.3f}s)")
            print(f"   Sistema Transformer: {Fore.RED if resultado_transformer else Fore.GREEN}{'✓' if not resultado_transformer else '✗'}{Style.RESET_ALL} ({tempo_transformer:.3f}s)")
        
        # Resumo para este caso
        print(f"\n{Fore.YELLOW}Resumo para {caso['categoria']}:{Style.RESET_ALL}")
        print(f"  Sistema baseado em regras:")
        print(f"    - Acertou {corretas_regras}/{len(caso['variacoes'])} variações corretas ({corretas_regras/len(caso['variacoes'])*100:.1f}%)")
        print(f"    - Rejeitou {len(caso['incorretas'])-falsos_positivos_regras}/{len(caso['incorretas'])} respostas incorretas ({(len(caso['incorretas'])-falsos_positivos_regras)/len(caso['incorretas'])*100:.1f}%)")
        
        print(f"  Sistema Transformer:")
        print(f"    - Acertou {corretas_transformer}/{len(caso['variacoes'])} variações corretas ({corretas_transformer/len(caso['variacoes'])*100:.1f}%)")
        print(f"    - Rejeitou {len(caso['incorretas'])-falsos_positivos_transformer}/{len(caso['incorretas'])} respostas incorretas ({(len(caso['incorretas'])-falsos_positivos_transformer)/len(caso['incorretas'])*100:.1f}%)")
        
        # Acumula resultados globais
        total_corretas_regras += corretas_regras
        total_corretas_transformer += corretas_transformer
        total_variacoes += len(caso['variacoes'])
        total_incorretas_regras += len(caso['incorretas']) - falsos_positivos_regras
        total_incorretas_transformer += len(caso['incorretas']) - falsos_positivos_transformer
        total_incorretas += len(caso['incorretas'])
        
        print(f"{Fore.CYAN}{'-'*100}{Style.RESET_ALL}")
    
    # Resumo global
    print(f"\n{Fore.CYAN}{'='*100}")
    print(f" RESUMO GLOBAL")
    print(f"{'='*100}{Style.RESET_ALL}")
    
    print(f"{Fore.YELLOW}Sistema baseado em regras:{Style.RESET_ALL}")
    print(f"  - Taxa de acerto para respostas corretas: {total_corretas_regras}/{total_variacoes} ({total_corretas_regras/total_variacoes*100:.1f}%)")
    print(f"  - Taxa de rejeição para respostas incorretas: {total_incorretas_regras}/{total_incorretas} ({total_incorretas_regras/total_incorretas*100:.1f}%)")
    print(f"  - Precisão geral: {(total_corretas_regras+total_incorretas_regras)/(total_variacoes+total_incorretas)*100:.1f}%")
    
    print(f"\n{Fore.YELLOW}Sistema Transformer:{Style.RESET_ALL}")
    print(f"  - Taxa de acerto para respostas corretas: {total_corretas_transformer}/{total_variacoes} ({total_corretas_transformer/total_variacoes*100:.1f}%)")
    print(f"  - Taxa de rejeição para respostas incorretas: {total_incorretas_transformer}/{total_incorretas} ({total_incorretas_transformer/total_incorretas*100:.1f}%)")
    print(f"  - Precisão geral: {(total_corretas_transformer+total_incorretas_transformer)/(total_variacoes+total_incorretas)*100:.1f}%")
    
    # Conclusão
    melhoria = ((total_corretas_transformer+total_incorretas_transformer)/(total_variacoes+total_incorretas)) - ((total_corretas_regras+total_incorretas_regras)/(total_variacoes+total_incorretas))
    
    print(f"\n{Fore.GREEN}CONCLUSÃO:{Style.RESET_ALL}")
    if melhoria > 0:
        print(f"O sistema baseado em Transformer apresentou uma melhoria de {melhoria*100:.1f}% na precisão geral.")
        print(f"Principalmente na capacidade de reconhecer variações corretas (+{(total_corretas_transformer-total_corretas_regras)/total_variacoes*100:.1f}%).")
    else:
        print(f"Neste conjunto de exemplos, o sistema baseado em regras teve desempenho superior.")
        print(f"Isso pode ocorrer em casos específicos onde as regras foram otimizadas manualmente.")

    print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")

if __name__ == "__main__":
    if not transformer_disponivel:
        print(f"{Fore.YELLOW}Para instalar as dependências necessárias execute:{Style.RESET_ALL}")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    comparar_sistemas(CASOS_TESTE) 