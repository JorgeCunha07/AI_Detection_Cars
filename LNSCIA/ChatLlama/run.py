import os
import torch

# https://www.llama.com/docs/how-to-guides/fine-tuning/

blob_path = r"C:\Users\CarlosMoutinho(11408\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa"

if os.path.isfile(blob_path):
    print("O arquivo blob existe.")
    try:
        # Força o carregamento com weights_only=False
        state_dict = torch.load(blob_path, map_location="cuda", weights_only=False)
        print("State dict carregado com sucesso.")
        print("Chaves do state dict:", list(state_dict.keys()))
    except Exception as e:
        print("Erro ao carregar o arquivo blob:", e)
else:
    print("Arquivo blob não encontrado.")
