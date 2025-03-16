A sequência recomendada para executar o projeto é a seguinte:

Pré-processamento dos Dados:
Execute o arquivo src/data_preparation.py (ou utilize o notebook notebooks/1_data_preparation.ipynb) para preparar e salvar os dados (tokenizers, sequências de entrada/saída, vocabulários, etc.).

Treinamento do Modelo:
Com os dados preparados, execute src/train.py para treinar o modelo Seq2Seq com atenção. Esse script irá compilar o modelo, treinar com os dados pré-processados e salvar os pesos em outputs/model_weights/.

Inferência (Geração):
Após o treinamento, use src/inference.py (ou o notebook notebooks/3_inference_demo.ipynb) para carregar os modelos de inferência e testar a geração de descrições para novas entradas.

Essa ordem garante que os dados estejam preparados antes de treinar e, com os pesos salvos, você pode realizar a inferência.