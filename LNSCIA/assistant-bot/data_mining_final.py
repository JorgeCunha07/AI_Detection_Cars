import json
import requests
import time
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
            
            # Verificar se a página existe
            if "Página não encontrada" in resposta.text:
                print(f"Página não encontrada para a questão {i}")
                continue
            
            # Usar BeautifulSoup para analisar o HTML
            soup = BeautifulSoup(resposta.text, 'html.parser')
            
            # Extrair a pergunta - exatamente como está no HTML que você compartilhou
            pergunta_div = soup.select_one('.question-text .text')
            if not pergunta_div:
                print(f"Pergunta não encontrada na questão {i}")
                continue
            
            pergunta = pergunta_div.get_text(strip=True)
            
            # Extrair as respostas - usando a estrutura exata que você compartilhou
            respostas_ul = soup.select_one('ul.answers')
            if not respostas_ul:
                print(f"Lista de respostas não encontrada na questão {i}")
                continue
            
            # Encontrar a resposta correta - procurando pela classe "correct" no li
            resposta_correta_li = respostas_ul.select_one('li.correct')
            if not resposta_correta_li:
                # Tentar outra abordagem se não encontrar diretamente
                resposta_correta_li = respostas_ul.select_one('li.answer.correct')
            
            if not resposta_correta_li:
                print(f"Resposta correta não encontrada na questão {i}")
                continue
            
            # Extrair a opção (A, B, C, etc.)
            opcao_span = resposta_correta_li.select_one('span.option')
            if not opcao_span:
                print(f"Opção da resposta correta não encontrada na questão {i}")
                continue
            
            opcao_correta = opcao_span.get_text(strip=True)
            
            # Extrair o texto da resposta
            texto_span = resposta_correta_li.select_one('span.answer-text')
            if not texto_span:
                print(f"Texto da resposta correta não encontrado na questão {i}")
                continue
            
            resposta_correta = texto_span.get_text(strip=True)
            
            # Extrair todas as opções
            todas_opcoes = {}
            for li in respostas_ul.select('li.answer'):
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

# Função para testar a extração em uma questão específica
def testar_extracao(numero_questao):
    url = f"https://www.bomcondutor.pt/questao/{numero_questao}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    resposta = requests.get(url, headers=headers)
    resposta.encoding = 'utf-8'
    
    soup = BeautifulSoup(resposta.text, 'html.parser')
    
    print(f"Testando extração para a questão {numero_questao}:")
    
    # Extrair a pergunta
    pergunta_div = soup.select_one('.question-text .text')
    if pergunta_div:
        pergunta = pergunta_div.get_text(strip=True)
        print(f"Pergunta encontrada: {pergunta}")
    else:
        print("Pergunta não encontrada")
    
    # Extrair as respostas
    respostas_ul = soup.select_one('ul.answers')
    if respostas_ul:
        print("Lista de respostas encontrada")
        
        # Encontrar todos os itens de resposta
        respostas_li = respostas_ul.select('li.answer')
        print(f"Encontradas {len(respostas_li)} opções de resposta")
        
        # Mostrar todas as opções
        for li in respostas_li:
            opcao = li.select_one('span.option')
            texto = li.select_one('span.answer-text')
            is_correct = 'correct' in li.get('class', [])
            
            if opcao and texto:
                print(f"Opção {opcao.get_text(strip=True)}: {texto.get_text(strip=True)} {'(CORRETA)' if is_correct else ''}")
        
        # Encontrar a resposta correta
        resposta_correta_li = respostas_ul.select_one('li.correct')
        if not resposta_correta_li:
            resposta_correta_li = respostas_ul.select_one('li.answer.correct')
        
        if resposta_correta_li:
            opcao = resposta_correta_li.select_one('span.option')
            texto = resposta_correta_li.select_one('span.answer-text')
            
            if opcao and texto:
                print(f"\nResposta correta identificada: {opcao.get_text(strip=True)} - {texto.get_text(strip=True)}")
            else:
                print("\nResposta correta encontrada, mas não foi possível extrair opção ou texto")
        else:
            print("\nNenhuma resposta correta encontrada")
    else:
        print("Lista de respostas não encontrada")

if __name__ == "__main__":
    # Testar a extração em uma questão específica
    print("Testando extração na questão 2...")
    testar_extracao(2)
    
    # Perguntar se deseja continuar com a extração completa
    print("\nDeseja continuar com a extração completa? (s/n): ", end="")
    resposta = input().strip().lower()
    if resposta == 's':
        print("Iniciando extração completa...")
        corpus = criar_corpus_bomcondutor(1, 50)
        
        # Integrar com o chatbot
        print("\nDeseja integrar este corpus ao conhecimento do chatbot? (s/n): ", end="")
        resposta = input().strip().lower()
        if resposta == 's':
            from criar_corpus_codigo_estrada import integrar_corpus_ao_chatbot
            integrar_corpus_ao_chatbot("corpus_bomcondutor.json") 