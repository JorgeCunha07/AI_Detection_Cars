import json
import time
import os
import io
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# Configurações de diretórios e caminhos
IMAGES_DIR = "./imagens/"  # Altere para o diretório desejado para as imagens
JSON_FILE = "./json/perguntas_respostas_bomcondutor.json"  # Altere para o caminho desejado para o JSON

# Garantir que os diretórios existam
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(JSON_FILE), exist_ok=True)

def configurar_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--log-level=3")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def descrever_imagem(url_imagem):
    """
    Utiliza uma API de IA para gerar uma descrição visual da imagem.
    Baixa a imagem e a envia como arquivo na requisição.
    """
    api_url = "https://api.deepai.org/api/image-captioning"
    headers = {"api-key": "cf64210a-f063-473c-b633-7feee16822e2"}  # Substitua pela sua chave de API
    try:
        # Baixa a imagem e lê seu conteúdo
        img_response = requests.get(url_imagem)
        img_response.raise_for_status()
        img_bytes = io.BytesIO(img_response.content)
        # Envia a imagem para a API como arquivo
        files = {'image': ('image.jpg', img_bytes)}
        response = requests.post(api_url, files=files, headers=headers)
        response.raise_for_status()
        result = response.json()
        descricao = result.get("output", "Descrição não encontrada")
        return descricao
    except Exception as e:
        return f"Erro ao descrever imagem: {e}"


def salvar_imagem(url_imagem, nome_arquivo):
    """
    Faz o download da imagem a partir da URL e salva localmente com o caminho especificado.
    """
    try:
        response = requests.get(url_imagem, stream=True)
        response.raise_for_status()
        with open(nome_arquivo, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Imagem salva em: {nome_arquivo}")
    except Exception as e:
        print(f"Erro ao salvar imagem: {e}")

def extrair_explicacao(soup):
    """
    Extrai a explicação da questão, retornando um dicionário com as chaves:
      - "texto": o texto principal da explicação (excluindo os materiais, se houver)
      - "materiais": uma lista de materiais de estudo, onde cada material é um dicionário com "texto" e "link"
    Se não houver explicação relevante, retorna uma string vazia.
    """
    explicacao_obj = {"texto": "", "materiais": []}
    conteiner = soup.select_one('div#explicacao .contents')
    if conteiner:
        resources = conteiner.select_one('div.resources')
        if resources:
            for li in resources.select('ul.study-materials li.material'):
                a = li.find('a')
                if a:
                    material = {
                        "texto": a.get_text(strip=True),
                        "link": a['href'] if a.has_attr('href') else ""
                    }
                    explicacao_obj["materiais"].append(material)
            resources.decompose()
        texto = conteiner.get_text(" ", strip=True)
        if "Esta questão ainda não possui conteúdo auxiliar." in texto:
            texto = ""
        explicacao_obj["texto"] = texto
    return explicacao_obj

def extrair_perguntas_respostas_bomcondutor(inicio=1, fim=100):
    dados = []
    driver = configurar_driver()

    for i in range(inicio, fim + 1):
        url = f"https://www.bomcondutor.pt/questao/{i}"
        print(f"Extraindo conteúdo de: {url}")
        try:
            driver.get(url)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            if "Página não encontrada" in html:
                print(f"Página não encontrada para a questão {i}")
                continue

            pergunta_div = soup.select_one('.question-text .text')
            if not pergunta_div:
                print(f"Pergunta não encontrada na questão {i}")
                continue
            pergunta = pergunta_div.get_text(strip=True)
            print(f"Pergunta: {pergunta}")

            respostas_html = soup.select('ul.answers li.answer')
            if not respostas_html:
                print(f"Nenhuma resposta encontrada na questão {i}")
                continue

            respostas = []
            for resp in respostas_html:
                option = resp.select_one('span.option')
                texto = resp.select_one('span.answer-text')
                if option and texto:
                    texto_resposta = texto.get_text(strip=True)
                    if "correct" in resp.get("class", []):
                        texto_resposta += " (correta)"
                    respostas.append(texto_resposta)
                    print(f"Resposta: {texto_resposta}")

            explicacao = extrair_explicacao(soup)
            print(f"Explicação: {explicacao}")

            item = {
                "id": i,
                "pergunta": pergunta,
                "respostas": respostas,
                "explicacao": explicacao
            }

            imagem = soup.select_one('.question-image img')
            if imagem and imagem.get('src'):
                src = imagem['src']
                if not src.startswith("http"):
                    src = "https://www.bomcondutor.pt" + src
                item["imagem_url"] = src
                print(f"Imagem: {src}")
                #descricao = descrever_imagem(src)
                #item["imagem_descricao"] = descricao
                #print(f"Descrição da imagem: {descricao}")
                
                ext = os.path.splitext(src)[1]
                if not ext:
                    ext = ".jpg"
                # Define o caminho completo para salvar a imagem
                nome_arquivo = os.path.join(IMAGES_DIR, f"imagem_questao_{i}{ext}")
                salvar_imagem(src, nome_arquivo)
                item["imagem_salva"] = nome_arquivo
            
            dados.append(item)
            print("-" * 50)
            
        except Exception as e:
            print(f"Erro ao processar a questão {i}: {str(e)}")
            continue

    driver.quit()

    # Carregar dados existentes e salvar no JSON usando o caminho configurado
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            dados_existentes = json.load(f)
            if not isinstance(dados_existentes, list):
                dados_existentes = []
    except FileNotFoundError:
        dados_existentes = []

    dados_final = dados_existentes + dados

    with open(JSON_FILE, "w", encoding="utf-8") as arquivo:
        json.dump(dados_final, arquivo, ensure_ascii=False, indent=4)
    
    print(f"Extração concluída! Total de {len(dados_final)} questões no arquivo.")
    return dados_final

def salvar_html_questao(numero_questao):
    url = f"https://www.bomcondutor.pt/questao/{numero_questao}"
    driver = configurar_driver()
    try:
        driver.get(url)
        time.sleep(3)
        html = driver.page_source
        nome_html = f"questao_{numero_questao}.html"
        with open(nome_html, "w", encoding="utf-8") as arquivo:
            arquivo.write(html)
        print(f"HTML da questão {numero_questao} salvo com sucesso em: {nome_html}")
    except Exception as e:
        print(f"Erro ao salvar HTML da questão {numero_questao}: {str(e)}")
    finally:
        driver.quit()

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
