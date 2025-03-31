import json

def converter_codigo_estrada(data):
    artigos_convertidos = []
    id_counter = 1

    # Percorre cada título na lista "conteudo"
    for titulo in data.get("conteudo", []):
        title = titulo.get("titulo", "")
        # Percorre cada capítulo do título
        for capitulo in titulo.get("capitulos", []):
            chapter = capitulo.get("capitulo", "")
            # Percorre os artigos dentro do capítulo
            for artigo in capitulo.get("artigos", []):
                article_num = artigo.get("artigo", "")
                text = artigo.get("conteudo", "")
                
                # Define o contexto como os primeiros 100 caracteres do texto (pode ser modificado)
                context = text[:100] + "..." if len(text) > 100 else text
                
                # Cria a referência automaticamente
                reference = f"Código da Estrada, {title}, {chapter}, {article_num}"
                
                artigo_convertido = {
                    "id": str(id_counter),
                    "title": title,
                    "chapter": chapter,
                    "article": article_num,
                    "context": context,
                    "text": text,
                    "reference": reference
                }
                artigos_convertidos.append(artigo_convertido)
                id_counter += 1

    return {"articles": artigos_convertidos}

def main():
    # Lê o ficheiro JSON original
    with open("Codigo_Estrada.json", "r", encoding="utf-8") as infile:
        data = json.load(infile)

    # Converte os dados para o novo formato
    dados_convertidos = converter_codigo_estrada(data)

    # Seleciona os primeiros 20 artigos
    primeiros_20 = {"articles": dados_convertidos["articles"][:20]}

    # Salva o resultado completo num novo ficheiro JSON
    with open("Codigo_Estrada_converted.json", "w", encoding="utf-8") as outfile:
        json.dump(dados_convertidos, outfile, ensure_ascii=False, indent=2)

    # Salva os primeiros 20 artigos num ficheiro separado
    with open("Codigo_Estrada_first20.json", "w", encoding="utf-8") as outfile20:
        json.dump(primeiros_20, outfile20, ensure_ascii=False, indent=2)

    # Imprime os primeiros 20 artigos no console
    print("Primeiros 20 artigos convertidos:")
    for artigo in primeiros_20["articles"]:
        print(json.dumps(artigo, ensure_ascii=False, indent=2))
    print("\nConversão concluída. Verifique os ficheiros 'Codigo_Estrada_converted.json' e 'Codigo_Estrada_first20.json'.")

if __name__ == "__main__":
    main()
