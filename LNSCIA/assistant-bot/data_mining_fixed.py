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

            soup = BeautifulSoup(resposta.text, 'html.parser')

            # Extraindo pergunta
            pergunta_html = soup.select_one('.question-text .text')
            if not pergunta_html:
                print(f"Pergunta não encontrada na questão {i}")
                continue
            pergunta = pergunta_html.get_text(strip=True)

            # Extraindo resposta correta - CORREÇÃO AQUI
            # Baseado na estrutura HTML que você compartilhou
            resposta_correta = None
            opcao_correta = None
            
            # Procurar por elementos li com classe 'answer' e 'correct'
            resposta_html = soup.select('li.answer.correct')
            if resposta_html:
                # Extrair a opção (A, B, C, etc.)
                opcao_span = resposta_html[0].select_one('span.option')
                if opcao_span:
                    opcao_correta = opcao_span.get_text(strip=True)
                
                # Extrair o texto da resposta
                texto_span = resposta_html[0].select_one('span.answer-text')
                if texto_span:
                    resposta_correta = texto_span.get_text(strip=True)
            
            # Se não encontrou usando o seletor acima, tentar outra abordagem
            if not resposta_correta:
                # Procurar por qualquer li que tenha a classe 'correct'
                for li in soup.select('li'):
                    if 'correct' in li.get('class', []):
                        # Extrair a opção
                        opcao_span = li.select_one('span.option')
                        if opcao_span:
                            opcao_correta = opcao_span.get_text(strip=True)
                        
                        # Extrair o texto da resposta
                        texto_span = li.select_one('span.answer-text')
                        if texto_span:
                            resposta_correta = texto_span.get_text(strip=True)
                        break
            
            if not resposta_correta:
                print(f"Resposta correta não encontrada na questão {i}")
                continue

            # Extraindo todas as opções para contexto completo
            todas_opcoes = {}
            for opcao_li in soup.select('li.answer'):
                opcao_letra = opcao_li.select_one('span.option')
                opcao_texto = opcao_li.select_one('span.answer-text')
                if opcao_letra and opcao_texto:
                    todas_opcoes[opcao_letra.get_text(strip=True)] = opcao_texto.get_text(strip=True)

            # Extraindo explicação (se disponível)
            explicacao_html = soup.select_one('div#explicacao .contents')
            explicacao = ""
            if explicacao_html:
                explicacao_texto = explicacao_html.get_text(strip=True)
                if "Esta questão ainda não possui conteúdo auxiliar." not in explicacao_texto:
                    explicacao = explicacao_texto

            # Criar padrão de regex para a pergunta
            palavras_chave = pergunta.lower().split()[:3]
            # Remover palavras muito comuns se necessário
            palavras_filtradas = [p for p in palavras_chave if len(p) > 3 and p not in ['esta', 'este', 'nesta', 'neste', 'devo', 'deve', 'para', 'como', 'qual', 'quais']]
            if palavras_filtradas:
                padrao_regex = r"(?i).*" + ".*".join(palavras_filtradas) + ".*"
            else:
                padrao_regex = r"(?i).*" + ".*".join(palavras_chave) + ".*"

            # Formatar a resposta de forma mais informativa
            resposta_formatada = f"A resposta correta é {opcao_correta}: {resposta_correta}"
            
            # Adicionar contexto das outras opções se disponível
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
            if todas_opcoes:
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

if __name__ == "__main__":
    # Começar com um número menor para testar
    corpus = criar_corpus_bomcondutor(1, 100)
    
    # Integrar com o chatbot
    print("\nDeseja integrar este corpus ao conhecimento do chatbot? (s/n): ", end="")
    resposta = input().strip().lower()
    if resposta == 's':
        from criar_corpus_codigo_estrada import integrar_corpus_ao_chatbot
        integrar_corpus_ao_chatbot("corpus_bomcondutor.json") 