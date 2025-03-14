import json

# Carregar o arquivo original
with open('../Documentacao/BomCondutor/json/perguntas_respostas_bomcondutor.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

new_data = []

# Processar cada item do arquivo
for entry in data:
    question = entry.get('pergunta', '').strip()
    respostas = entry.get('respostas', [])
    
    # Buscar a resposta correta (aquela que contém "(correta)")
    correct_answer = None
    for resposta in respostas:
        if "(correta)" in resposta:
            # Remover a marcação "(correta)" e limpar espaços
            correct_answer = resposta.replace("(correta)", "").strip()
            break
    # Se não encontrar a resposta correta, usar a primeira resposta disponível
    if correct_answer is None and respostas:
        correct_answer = respostas[0].strip()
    
    # Criar novo objeto para o corpus
    new_entry = {
        "input": question,
        "target": correct_answer
    }
    new_data.append(new_entry)

# Salvar o novo JSON
with open('corpus_integration.json', 'w', encoding='utf-8') as outfile:
    json.dump(new_data, outfile, indent=4, ensure_ascii=False)

print("Novo JSON criado com sucesso.")
