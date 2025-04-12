import json
import torch

from pathlib import Path
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Caminho para questoes.json
base_dir = Path(__file__).resolve().parent
questoes_path = base_dir / "questoes.json"

# Carregar questões
with open(questoes_path, encoding="utf-8") as f:
    questoes = json.load(f)

# Modelos carregados uma vez
sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
transformer_model_name = "cross-encoder/stsb-roberta-base"
transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_model_name)
transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

# Limpeza básica
def limpar_texto(texto):
    return texto.lower().strip()

# Similaridade simples com difflib
def similaridade_basica(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Função principal para validação da resposta
def validar_resposta(pergunta, resposta_usuario, metodo_semantico="sbert"):
    pergunta_limpa = limpar_texto(pergunta)
    resposta_limpa = limpar_texto(resposta_usuario)

    for q in questoes:
        if limpar_texto(q["pergunta"]) == pergunta_limpa:
            corretas = q.get("respostas_corretas", [])
            # incorretas = q.get("respostas_incorretas", [])
            corretas_limpas = [limpar_texto(r) for r in corretas]

            # Verificação de match direto
            if resposta_limpa in corretas_limpas:
                return {"correto": True, "metodo": "match direto"}

            # Verificação usando similaridade básica
            for r in corretas_limpas:
                if similaridade_basica(r, resposta_limpa) > 0.85:
                    return {"correto": True, "metodo": "similaridade basica"}

            # Método semântico SBERT
            if metodo_semantico == "sbert":
                print("Usando sbert para validação...")
                emb_user = sbert_model.encode(resposta_usuario, convert_to_tensor=True)
                emb_refs = sbert_model.encode(corretas, convert_to_tensor=True)
                score = float(util.cos_sim(emb_user, emb_refs).max())
                if score > 0.75:
                    return {"correto": True, "metodo": "SBERT", "score": score}

            # Método semântico com Transformer (usando saída de regressão)
            elif metodo_semantico == "transformer":
                print("Usando Transformer para validação...")
                transformer_model.eval()
                scores = []
                for ref in corretas:
                    inputs = transformer_tokenizer.encode_plus(
                        resposta_usuario, ref, return_tensors="pt", truncation=True
                    ).to(transformer_model.device)
                    with torch.no_grad():
                        logits = transformer_model(**inputs).logits
                    # Usando a saída de regressão diretamente
                    score = logits[0][0].item()
                    scores.append(score)

                max_score = max(scores)
                if max_score > 0.75:
                    return {"correto": True, "metodo": "Transformer pipeline", "score": max_score}

            return {"correto": False, "respostas_aceites": corretas}

    return {"erro": "Pergunta não encontrada no banco de dados."}
