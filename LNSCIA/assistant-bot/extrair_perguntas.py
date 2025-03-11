import json
import requests
import time
from bs4 import BeautifulSoup

def extrair_perguntas_bomcondutor(inicio=1, fim=100):
    perguntas = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for i in range(inicio, fim + 1):
        url = f"https://www.bomcondutor.pt/questao/{i}"
        print(f"Extraindo pergunta de: {url}")

        try:
            # Adicionar um pequeno atraso para não sobrecarregar o servidor
            time.sleep(0.5)
            
            resposta = requests.get(url, headers=headers)
            resposta.encoding = 'utf-8'
            
            # Verificar se a página existe
            if "Página não encontrada" in resposta.text:
                print(f"Página não encontrada para a questão {i}")
                continue
            
            # Usar BeautifulSoup para analisar o HTML
            soup = BeautifulSoup(resposta.text, 'html.parser')
            
            # Extrair a pergunta
            pergunta_div = soup.select_one('.question-text .text')
            if not pergunta_div:
                print(f"Pergunta não encontrada na questão {i}")
                continue
            
            pergunta = pergunta_div.get_text(strip=True)
            
            # Adicionar a pergunta à lista
            perguntas.append({
                "id": i,
                "pergunta": pergunta
            })
            
            print(f"Pergunta extraída: {pergunta}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Erro ao processar a questão {i}: {str(e)}")
            # Continuar com a próxima questão em caso de erro
            continue
    
    # Salvar as perguntas em um arquivo JSON
    with open("perguntas_bomcondutor.json", "w", encoding="utf-8") as arquivo:
        json.dump(perguntas, arquivo, ensure_ascii=False, indent=4)
    
    print(f"Extração concluída! Total de {len(perguntas)} perguntas extraídas.")
    return perguntas

# Função para criar um corpus simples com as perguntas
def criar_corpus_simples(perguntas):
    corpus = {"perguntas_respostas": []}
    
    for item in perguntas:
        pergunta = item["pergunta"]
        
        # Criar padrão de regex para a pergunta
        palavras_chave = [p for p in pergunta.lower().split() if len(p) > 3 and p not in ['esta', 'este', 'nesta', 'neste', 'devo', 'deve', 'para', 'como', 'qual', 'quais']]
        if len(palavras_chave) >= 3:
            palavras_chave = palavras_chave[:3]
        elif len(palavras_chave) == 0:
            palavras_chave = pergunta.lower().split()[:3]
        
        padrao_regex = r"(?i).*" + ".*".join(palavras_chave) + ".*"
        
        corpus_entry = {
            "padrao": padrao_regex,
            "respostas": [
                f"De acordo com o Código da Estrada: {pergunta}",
                f"Esta pergunta está relacionada ao Código da Estrada. Para ver a resposta completa, visite: https://www.bomcondutor.pt/questao/{item['id']}"
            ]
        }
        
        corpus["perguntas_respostas"].append(corpus_entry)
    
    # Salvar o corpus em um arquivo JSON
    with open("corpus_bomcondutor_simples.json", "w", encoding="utf-8") as arquivo:
        json.dump(corpus, arquivo, ensure_ascii=False, indent=4)
    
    print(f"Corpus simples criado com sucesso! Total de {len(corpus['perguntas_respostas'])} entradas.")
    return corpus

if __name__ == "__main__":
    print("Iniciando extração de perguntas...")
    perguntas = extrair_perguntas_bomcondutor(1, 1000)  # Extrair 1000 perguntas
    
    print("\nDeseja criar um corpus simples com as perguntas extraídas? (s/n): ", end="")
    resposta = input().strip().lower()
    if resposta == 's':
        corpus = criar_corpus_simples(perguntas)
        
        # Integrar com o chatbot
        print("\nDeseja integrar este corpus ao conhecimento do chatbot? (s/n): ", end="")
        resposta = input().strip().lower()
        if resposta == 's':
            from criar_corpus_codigo_estrada import integrar_corpus_ao_chatbot
            integrar_corpus_ao_chatbot("corpus_bomcondutor_simples.json") 