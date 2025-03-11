import json
import requests
import time
import re
from bs4 import BeautifulSoup

def criar_corpus_bomcondutor(inicio=1, fim=100):
    corpus = {"perguntas_respostas": []}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for i in range(inicio, fim + 1):
        url = f"https://www.bomcondutor.pt/questao/{i}"
        print(f"Extraindo conteúdo de: {url}")

        try:
            # Adicionar um pequeno atraso para não sobrecarregar o servidor
            time.sleep(1)
            
            resposta = requests.get(url, headers=headers)
            resposta.encoding = 'utf-8'
            
            soup = BeautifulSoup(resposta.text, 'html.parser')
            
            # Extrair a pergunta
            pergunta_element = soup.select_one('.question-text .text')
            if not pergunta_element:
                print(f"Pergunta não encontrada na questão {i}")
                continue
                
            pergunta = pergunta_element.get_text(strip=True)
            
            # Encontrar a lista de respostas (ul class="answers")
            respostas_ul = soup.select_one('ul.answers')
            if not respostas_ul:
                print(f"Lista de respostas não encontrada na questão {i}")
                continue
                
            # Encontrar todos os itens de resposta (li class="answer")
            respostas_li = respostas_ul.select('li.answer')
            if not respostas_li:
                print(f"Itens de resposta não encontrados na questão {i}")
                continue
                
            # Procurar pelo item que contém a classe "correct"
            resposta_correta_li = None
            for li in respostas_li:
                if 'correct' in li.get('class', []):
                    resposta_correta_li = li
                    break
                    
            if not resposta_correta_li:
                print(f"Resposta correta não encontrada na questão {i}")
                continue
                
            # Extrair a letra da opção correta
            opcao_span = resposta_correta_li.select_one('span.option')
            if not opcao_span:
                print(f"Opção da resposta correta não encontrada na questão {i}")
                continue
                
            opcao_correta = opcao_span.get_text(strip=True)
            
            # Extrair o texto da resposta correta
            texto_span = resposta_correta_li.select_one('span.answer-text')
            if not texto_span:
                print(f"Texto da resposta correta não encontrado na questão {i}")
                continue
                
            resposta_correta = texto_span.get_text(strip=True)
            
            # Extrair todas as opções para contexto
            todas_opcoes = {}
            for li in respostas_li:
                opcao = li.select_one('span.option')
                texto = li.select_one('span.answer-text')
                if opcao and texto:
                    todas_opcoes[opcao.get_text(strip=True)] = texto.get_text(strip=True)
            
            # Extrair explicação (se disponível)
            explicacao = ""
            explicacao_div = soup.select_one('div#explicacao .contents')
            if explicacao_div:
                explicacao_texto = explicacao_div.get_text(strip=True)
                if "Esta questão ainda não possui conteúdo auxiliar" not in explicacao_texto:
                    explicacao = explicacao_texto
            
            # Criar padrão de regex para a pergunta
            palavras_chave = [p for p in pergunta.lower().split() if len(p) > 3 and p not in ['esta', 'este', 'nesta', 'neste', 'devo', 'deve', 'para', 'como', 'qual', 'quais']]
            if len(palavras_chave) >= 3:
                palavras_chave = palavras_chave[:3]
            elif len(palavras_chave) == 0:
                palavras_chave = pergunta.lower().split()[:3]
                
            padrao_regex = r"(?i).*" + ".*".join(palavras_chave) + ".*"
            
            # Formatar a resposta
            resposta_formatada = f"A resposta correta é {opcao_correta}: {resposta_correta}"
            
            # Adicionar contexto das outras opções
            if todas_opcoes:
                contexto_opcoes = " | ".join([f"{letra}: {texto}" for letra, texto in todas_opcoes.items()])
                resposta_formatada += f"\nOpções: {contexto_opcoes}"
                
            corpus_entry = {
                "padrao": padrao_regex,
                "respostas": [
                    f"De acordo com o Código da Estrada: {pergunta} {resposta_formatada}"
                ],
            }
            
            if explicacao:
                corpus_entry["respostas"].append(f"Explicação: {explicacao}")
                
            print(f"Pergunta: {pergunta}")
            print(f"Resposta correta: {opcao_correta} - {resposta_correta}")
            print(f"Todas as opções: {todas_opcoes}")
            if explicacao:
                print(f"Explicação: {explicacao}")
            print("-" * 50)
            
            corpus["perguntas_respostas"].append(corpus_entry)
            
        except Exception as e:
            print(f"Erro ao processar a questão {i}: {str(e)}")
            # Continuar com a próxima questão em caso de erro
            continue
            
    with open("corpus_bomcondutor.json", "w", encoding="utf-8") as arquivo:
        json.dump(corpus, arquivo, ensure_ascii=False, indent=4)
        
    print(f"Corpus criado com sucesso! Total de {len(corpus['perguntas_respostas'])} perguntas adicionadas.")
    return corpus

# Função para analisar uma questão específica e mostrar a estrutura HTML relevante
def analisar_questao(numero_questao):
    url = f"https://www.bomcondutor.pt/questao/{numero_questao}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    resposta = requests.get(url, headers=headers)
    resposta.encoding = 'utf-8'
    
    soup = BeautifulSoup(resposta.text, 'html.parser')
    
    # Encontrar a lista de respostas
    respostas_ul = soup.select_one('ul.answers')
    if respostas_ul:
        print("Estrutura da lista de respostas:")
        print(respostas_ul.prettify())
        
        # Encontrar o item correto
        for li in respostas_ul.select('li'):
            if 'correct' in li.get('class', []):
                print("\nResposta correta encontrada:")
                print(li.prettify())
                
                opcao = li.select_one('span.option')
                texto = li.select_one('span.answer-text')
                if opcao and texto:
                    print(f"\nOpção correta: {opcao.get_text(strip=True)}")
                    print(f"Texto da resposta: {texto.get_text(strip=True)}")
                break
        else:
            print("Nenhum item com a classe 'correct' foi encontrado.")
    else:
        print("Nenhuma lista de respostas (ul.answers) encontrada.")

if __name__ == "__main__":
    # Opção para analisar uma questão específica
    print("Deseja analisar a estrutura HTML de uma questão específica? (s/n): ", end="")
    resposta = input().strip().lower()
    if resposta == 's':
        numero = int(input("Digite o número da questão: "))
        analisar_questao(numero)
    
    # Começar a extração
    print("Iniciando extração de dados...")
    corpus = criar_corpus_bomcondutor(1, 50)
    
    # Integrar com o chatbot
    print("\nDeseja integrar este corpus ao conhecimento do chatbot? (s/n): ", end="")
    resposta = input().strip().lower()
    if resposta == 's':
        from criar_corpus_codigo_estrada import integrar_corpus_ao_chatbot
        integrar_corpus_ao_chatbot("corpus_bomcondutor.json") 