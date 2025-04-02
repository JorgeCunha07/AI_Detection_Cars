# io_utils.py
import json
from colorama import Fore, Style

def load_training_data(codigo_file: str, questions_file: str):
    print(f"{Fore.CYAN}Carregando dados de treino...{Style.RESET_ALL}")
    with open(codigo_file, encoding="utf-8") as f:
        codigo_data = json.load(f)
    with open(questions_file, encoding="utf-8") as f:
        questions_data = json.load(f)
    
    examples = []
    # Cria exemplos a partir dos artigos do Código da Estrada
    for article in codigo_data.get("articles", []):
        prompt = f"Explique o seguinte artigo do Código da Estrada: {article['article']}\nResposta: {article['text']}"
        examples.append({"text": prompt})
    # Cria exemplos a partir das perguntas e respostas
    for item in questions_data:
        for resposta in item.get("respostas_corretas", []):
            prompt = f"Pergunta: {item['pergunta']}\nResposta: {resposta}"
            examples.append({"text": prompt})
    print(f"{Fore.GREEN}Dados carregados: {len(examples)} exemplos{Style.RESET_ALL}")
    return examples

def load_quiz_data(file_path: str = "../../Documentacao/Questoes/questoes.json"):
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Erro ao carregar perguntas: {e}{Style.RESET_ALL}")
        return []

def load_codigo_data(file_path: str = "../../Documentacao/BomCondutor/Codigo_Estrada.json"):
    try:
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Erro ao carregar Código da Estrada: {e}{Style.RESET_ALL}")
        return {"articles": []}
