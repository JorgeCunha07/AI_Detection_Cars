import os
from transformers import pipeline

# Monta o caminho absoluto para a pasta do modelo, subindo um nível a partir de src/
model_path = os.path.abspath(os.path.join("..", "fine_tuning_model"))

qa_pipeline = pipeline(
    "question-answering",
    model=model_path,       # caminho local para o modelo
    tokenizer=model_path,   # caminho local para o tokenizer
    local_files_only=True   # força a usar somente arquivos locais
)

result = qa_pipeline(
    question="Qual é a velocidade máxima em zonas urbanas?",
    context=(
        "De acordo com o Código da Estrada, a velocidade máxima permitida em zonas "
        "urbanas é de 50 km/h. Já em autoestradas, pode chegar a 120 km/h."
    )
)
print(result)

