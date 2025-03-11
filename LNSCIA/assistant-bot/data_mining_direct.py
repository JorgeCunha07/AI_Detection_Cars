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
            
            # Verificar se a página existe
            if "Página não encontrada" in resposta.text:
                print(f"Página não encontrada para a questão {i}")
                continue

            # Extrair o HTML bruto para análise
            html_content = resposta.text
            
            # Extrair a pergunta usando regex
            pergunta_match = re.search(r'<div class="text">(.*?)</div>', html_content, re.DOTALL)
            if not pergunta_match:
                print(f"Pergunta não encontrada na questão {i}")
                continue
                
            pergunta = pergunta_match.group(1).strip()
            # Limpar tags HTML da pergunta
            pergunta = re.sub(r'<.*?>', '', pergunta).strip()
            
            # Extrair as respostas e identificar a correta
            respostas_html = re.findall(r'<li class="answer ([A-Z])(?: correct)?">\s*<span class="option">([A-Z])</span>\s*<span class="answer-text">(.*?)</span>', html_content)
            
            if not respostas_html:
                # Tentar outro padrão
                respostas_html = re.findall(r'<li class="answer ([A-Z])(?: correct)?"><span class="option">([A-Z])</span><span class="answer-text">(.*?)</span>', html_content)
            
            if not respostas_html:
                print(f"Respostas não encontradas na questão {i}")
                continue
                
            # Procurar pela resposta correta
            resposta_correta = None
            opcao_correta = None
            todas_opcoes = {}
            
            # Verificar se há uma classe "correct" no HTML
            correct_match = re.search(r'<li class="answer ([A-Z]) correct">', html_content)
            if correct_match:
                opcao_correta = correct_match.group(1)
                
            # Processar todas as respostas
            for classe, letra, texto in respostas_html:
                todas_opcoes[letra] = texto.strip()
                if classe == opcao_correta or (opcao_correta is None and "correct" in html_content and letra in html_content + "correct"):
                    opcao_correta = letra
                    resposta_correta = texto.strip()
            
            if not resposta_correta and opcao_correta and opcao_correta in todas_opcoes:
                resposta_correta = todas_opcoes[opcao_correta]
                
            if not resposta_correta:
                print(f"Resposta correta não identificada na questão {i}")
                continue
                
            # Extrair explicação
            explicacao = ""
            explicacao_match = re.search(r'<div id="explicacao".*?<div class="contents">(.*?)</div>', html_content, re.DOTALL)
            if explicacao_match:
                explicacao_html = explicacao_match.group(1).strip()
                if "Esta questão ainda não possui conteúdo auxiliar" not in explicacao_html:
                    # Limpar tags HTML
                    explicacao = re.sub(r'<.*?>', '', explicacao_html).strip()
            
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

# Função para salvar o HTML para análise
def salvar_html_para_analise(numero_questao):
    url = f"https://www.bomcondutor.pt/questao/{numero_questao}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    resposta = requests.get(url, headers=headers)
    resposta.encoding = 'utf-8'
    
    with open(f"questao_{numero_questao}.html", "w", encoding="utf-8") as f:
        f.write(resposta.text)
    
    print(f"HTML da questão {numero_questao} salvo para análise.")

if __name__ == "__main__":
    # Opção para salvar o HTML de uma questão específica para análise
    print("Deseja salvar o HTML de uma questão específica para análise? (s/n): ", end="")
    resposta = input().strip().lower()
    if resposta == 's':
        numero = int(input("Digite o número da questão: "))
        salvar_html_para_analise(numero)
    
    # Começar com um número menor para testar
    print("Iniciando extração de dados...")
    corpus = criar_corpus_bomcondutor(1, 50)
    
    # Integrar com o chatbot
    print("\nDeseja integrar este corpus ao conhecimento do chatbot? (s/n): ", end="")
    resposta = input().strip().lower()
    if resposta == 's':
        from criar_corpus_codigo_estrada import integrar_corpus_ao_chatbot
        integrar_corpus_ao_chatbot("corpus_bomcondutor.json") 