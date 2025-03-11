import json
import requests
import time
from bs4 import BeautifulSoup

def extrair_perguntas_respostas_bomcondutor(inicio=1, fim=100):
    dados = []

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
            
            # Extrair a pergunta
            pergunta_div = soup.select_one('.question-text .text')
            if not pergunta_div:
                print(f"Pergunta não encontrada na questão {i}")
                continue
            
            pergunta = pergunta_div.get_text(strip=True)
            print(f"Pergunta: {pergunta}")
            
            # Extrair todas as respostas (sem se preocupar qual é a correta)
            respostas_html = soup.select('li.answer')
            if not respostas_html:
                print(f"Nenhuma resposta encontrada na questão {i}")
                continue
            
            respostas = []
            for resp in respostas_html:
                texto_resposta = resp.get_text(strip=True)
                # Remover "Certo." ou "Errado." do final
                texto_resposta = texto_resposta.replace('Certo.', '').replace('Errado.', '').strip()
                respostas.append(texto_resposta)
                print(f"Resposta: {texto_resposta}")
            
            # Salvar os dados
            item = {
                "id": i,
                "pergunta": pergunta,
                "respostas": respostas
                #"correta": respostaCorreta
            }
            
            # Verificar se há uma imagem na questão
            imagem = soup.select_one('.question-image img')
            if imagem and imagem.get('src'):
                item["imagem_url"] = imagem['src']
                print(f"Imagem: {imagem['src']}")
            
            dados.append(item)
            print("-" * 50)
            
        except Exception as e:
            print(f"Erro ao processar a questão {i}: {str(e)}")
            # Continuar com a próxima questão em caso de erro
            continue
    
    # Salvar os dados em um arquivo JSON
    with open("perguntas_respostas_bomcondutor.json", "w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=4)
    
    print(f"Extração concluída! Total de {len(dados)} questões extraídas.")
    return dados

# Função para salvar o HTML de uma questão específica para análise
def salvar_html_questao(numero_questao):
    url = f"https://www.bomcondutor.pt/questao/{numero_questao}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        resposta = requests.get(url, headers=headers)
        resposta.encoding = 'utf-8'
        
        with open(f"questao_{numero_questao}.html", "w", encoding="utf-8") as arquivo:
            arquivo.write(resposta.text)
        
        print(f"HTML da questão {numero_questao} salvo com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar HTML da questão {numero_questao}: {str(e)}")

if __name__ == "__main__":
    print("O que você deseja fazer?")
    print("1 - Extrair perguntas e respostas")
    print("2 - Salvar HTML de uma questão específica para análise")
    opcao = input("Digite a opção (1 ou 2): ").strip()
    
    if opcao == "1":
        inicio = int(input("Digite o número da primeira questão a extrair: ") or "1")
        fim = int(input("Digite o número da última questão a extrair: ") or "100")
        extrair_perguntas_respostas_bomcondutor(inicio, fim)
    elif opcao == "2":
        numero = int(input("Digite o número da questão para salvar o HTML: "))
        salvar_html_questao(numero)
    else:
        print("Opção inválida!") 