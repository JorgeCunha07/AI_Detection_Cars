import json
import requests
from bs4 import BeautifulSoup

def criar_corpus_bomcondutor(inicio=1, fim=10000):
    corpus = {"perguntas_respostas": []}
    headers = {'User-Agent': 'Mozilla/5.0'}

    for i in range(inicio, fim + 1):
        url = f"https://www.bomcondutor.pt/questao/{i}"
        print(f"Extraindo conteúdo de: {url}")

        resposta = requests.get(url, headers=headers)
        if resposta.status_code != 200:
            print(f"Erro ao acessar a questão {i}: Status code {resposta.status_code}")
            continue

        resposta.encoding = 'utf-8'
        soup = BeautifulSoup(resposta.text, 'html.parser')

        # Extrai a pergunta
        pergunta_element = soup.select_one('.question-text .text')
        if not pergunta_element:
            print(f"Pergunta não encontrada na questão {i}")
            continue
        pergunta = pergunta_element.get_text(strip=True)
        print(f"Pergunta: {pergunta}")

        # Extrai a resposta correta da <ul class="answers">
        ul_answers = soup.find("ul", class_="answers")
        resposta_correta = None
        if ul_answers:
            # Procura o <li> que contenha a classe "correct"
            li_correct = ul_answers.find("li", class_=lambda c: c and "correct" in c.split())
            if li_correct:
                span_text = li_correct.find("span", class_="answer-text")
                if span_text:
                    resposta_correta = span_text.get_text(strip=True)
        
        if not resposta_correta:
            print(f"Resposta correta não encontrada na questão {i}")
            continue

        print(f"Resposta correta: {resposta_correta}")

        # Extrai a explicação, se disponível
        explicacao_element = soup.select_one('div#explicacao .contents')
        explicacao = ""
        if explicacao_element:
            explicacao_texto = explicacao_element.get_text(strip=True)
            if "Esta questão ainda não possui conteúdo auxiliar." not in explicacao_texto:
                explicacao = explicacao_texto

        corpus_entry = {
            "padrao": pergunta,
            "respostas": [resposta_correta]
        }
        if explicacao:
            corpus_entry["explicacao"] = explicacao

        print("-" * 50)
        corpus["perguntas_respostas"].append(corpus_entry)

    with open("corpus_bomcondutor.json", "w", encoding="utf-8") as arquivo:
        json.dump(corpus, arquivo, ensure_ascii=False, indent=4)

    print(f"Corpus criado com sucesso! Total de {len(corpus['perguntas_respostas'])} perguntas adicionadas.")

if __name__ == "__main__":
    criar_corpus_bomcondutor(1, 10)
