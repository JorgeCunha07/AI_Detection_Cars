import re
import json

def ler_texto_txt(txt_file):
    """Lê o conteúdo do arquivo TXT."""
    with open(txt_file, "r", encoding="utf-8") as f:
        return f.read()

def limpar_texto(texto):
    """
    Limpa o texto removendo espaços extras e normalizando as quebras de linha.
    """
    # Substitui quebras de linha e múltiplos espaços por um único espaço
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def separar_artigos(texto):
    """
    Separa o texto em artigos com base nos cabeçalhos "Artigo X.º".
    Retorna uma lista de tuplas (header, content).
    """
    padrao = r'(Artigo\s+\d+\s*[\.º\u00BA])'
    partes = re.split(padrao, texto)
    
    lista_artigos = []
    # O cabeçalho dos artigos fica em posições ímpares e o conteúdo em posições pares subsequentes.
    for i in range(1, len(partes), 2):
        cabecalho = partes[i].strip()
        conteudo = partes[i+1].strip() if (i+1) < len(partes) else ""
        lista_artigos.append((cabecalho, conteudo))
    return lista_artigos

def processar_artigo(cabecalho, conteudo):
    """
    Processa um artigo separando o título do conteúdo.
    O título é considerado como a parte inicial do conteúdo, 
    opcionalmente precedida pelo caractere "º", até encontrar um padrão de início de cláusulas (por exemplo, "1 -").
    Retorna um dicionário com 'id', 'header', 'title' e 'content'.
    """
    # Extrai o número do artigo para gerar o id
    m = re.search(r'Artigo\s+(\d+)', cabecalho)
    article_id = f"Artigo_{m.group(1)}" if m else cabecalho.replace(" ", "_")
    
    # Padrão para capturar o título: opcional "º", seguido de qualquer texto até encontrar um número com " -"
    title_pattern = r'^(?:º\s*)?(.*?)\s*(?=\d+\s*-\s*)'
    m_title = re.search(title_pattern, conteudo)
    if m_title:
        title = m_title.group(1).strip()
        content_processed = conteudo[m_title.end():].strip()
    else:
        title = ""
        content_processed = conteudo
    return {
        "id": article_id,
        "header": cabecalho,
        "title": title,
        "content": content_processed
    }

def criar_dataset_json(articles, output_file):
    """
    Cria um dataset no formato JSON a partir da lista de artigos.
    Cada artigo é estruturado com 'id', 'header', 'title' e 'content'.
    """
    data = []
    for cabecalho, conteudo in articles:
        processed = processar_artigo(cabecalho, conteudo)
        data.append(processed)
    dataset = {"articles": data}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Dataset JSON salvo em '{output_file}'.")

if __name__ == "__main__":
    txt_file = "codigo_estrada.txt"                # Arquivo TXT com a informação manual
    output_json = "codigo_estrada_dataset.json"     # Arquivo JSON de saída

    # 1. Ler e limpar o texto do arquivo
    texto_bruto = ler_texto_txt(txt_file)
    texto_limpo = limpar_texto(texto_bruto)
    
    # 2. Separar o texto em artigos
    articles = separar_artigos(texto_limpo)
    print(f"Número de artigos encontrados: {len(articles)}")
    
    # 3. Processar e exibir um exemplo (primeiro artigo)
    if articles:
        primeiro = processar_artigo(*articles[0])
        print("\nExemplo do primeiro artigo:")
        print("Header:", primeiro["header"])
        print("Title:", primeiro["title"])
        print("Content (primeiros 300 caracteres):", primeiro["content"][:300])
    
    # 4. Criar e salvar o dataset JSON
    criar_dataset_json(articles, output_json)
