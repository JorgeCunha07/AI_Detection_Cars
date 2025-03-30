import re
import json
from datetime import date

def parse_codigo_estrada(file_path):
    # Cria a estrutura inicial com metadados e array de dados
    output = {
        "metadata": {
            "file_name": file_path,
            "description": "Trecho do Código da Estrada Português com definições, regras e sanções.",
            "source": "Código da Estrada / RHLC",
            "processed_date": str(date.today()),
            "version": "v1.0"
        },
        "data": []
    }
    
    current_title = None
    current_chapter = None

    # Abre o ficheiro com codificação UTF-8
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Ignora linhas em branco

            # Detecta linhas que começam com "Título"
            titulo_match = re.match(r"^(Título\s+.*)", line)
            if titulo_match:
                current_title = {
                    "titulo": titulo_match.group(1),
                    "capitulos": []
                }
                output["data"].append(current_title)
                current_chapter = None  # Reseta o capítulo atual
                continue

            # Detecta linhas que começam com "Capítulo"
            capitulo_match = re.match(r"^(Capítulo\s+.*)", line)
            if capitulo_match:
                current_chapter = {
                    "capitulo": capitulo_match.group(1),
                    "artigos": []
                }
                # Se não houver título atual, cria um título "Sem título"
                if current_title is None:
                    current_title = {"titulo": "Sem título", "capitulos": []}
                    output["data"].append(current_title)
                current_title["capitulos"].append(current_chapter)
                continue

            # Detecta linhas que começam com "Artigo"
            artigo_match = re.match(r"^(Artigo\s+\d+[.º\.\w]*)\s*—\s*(.*)", line)
            if artigo_match:
                artigo_title = artigo_match.group(1)
                artigo_content = artigo_match.group(2)
                artigo = {
                    "artigo": artigo_title,
                    "conteudo": artigo_content
                }
                # Se não houver capítulo atual, cria um capítulo "Sem capítulo"
                if current_chapter is None:
                    if current_title is None:
                        current_title = {"titulo": "Sem título", "capitulos": []}
                        output["data"].append(current_title)
                    current_chapter = {"capitulo": "Sem capítulo", "artigos": []}
                    current_title["capitulos"].append(current_chapter)
                current_chapter["artigos"].append(artigo)
                continue

            # Caso a linha não seja título, capítulo ou artigo, adiciona ao conteúdo do último artigo
            if current_chapter is not None and current_chapter["artigos"]:
                current_chapter["artigos"][-1]["conteudo"] += " " + line
            else:
                # Se não houver artigo, pode ser uma descrição extra para o capítulo ou título
                if current_chapter is not None:
                    current_chapter.setdefault("descricao", "")
                    current_chapter["descricao"] += " " + line
                elif current_title is not None:
                    current_title.setdefault("descricao", "")
                    current_title["descricao"] += " " + line

    return output

def main():
    input_file = "Codigo_Estrada.txt"  # Atualize com o caminho do seu ficheiro
    parsed_data = parse_codigo_estrada(input_file)
    
    # Salva o JSON no ficheiro de saída
    output_file = "Codigo_Estrada.json"
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(parsed_data, out_file, ensure_ascii=False, indent=4)
    print(f"JSON gerado com sucesso: {output_file}")

if __name__ == "__main__":
    main()
