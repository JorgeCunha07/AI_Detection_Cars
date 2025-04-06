import json
import random
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score

# Carregar os diálogos gold do arquivo
with open("dialogos_validos2.json", encoding="utf-8") as f:
    dialogos = json.load(f)

# Selecionar aleatoriamente 5 exemplos para os testes
exemplos = random.sample(dialogos, 5)

rouge_evaluator = Rouge()

bleu_scores = []
bert_f1_scores = []
rouge_1_scores = []
rouge_2_scores = []
rouge_l_scores = []

for exemplo in exemplos:
    # Obter o input e a resposta de referência
    entrada = exemplo["input"]
    referencia = exemplo["output"]

    # Para este teste, vamos assumir que o output gerado é igual à referência.
    # Em um cenário real, substituiríamos 'gerado' pela resposta obtida do modelo.
    gerado = referencia  

    # Cálculo do BLEU
    bleu = sentence_bleu([referencia.split()], gerado.split())
    bleu_scores.append(bleu)

    # Cálculo do ROUGE
    rouge_score = rouge_evaluator.get_scores(gerado, referencia)[0]
    rouge_1_scores.append(rouge_score["rouge-1"]["f"])
    rouge_2_scores.append(rouge_score["rouge-2"]["f"])
    rouge_l_scores.append(rouge_score["rouge-l"]["f"])

    # Cálculo do BERTScore
    P, R, F1 = score([gerado], [referencia], lang="pt", model_type="neuralmind/bert-base-portuguese-cased")
    bert_f1 = F1[0].item()
    bert_f1_scores.append(bert_f1)

    print("Input:", entrada)
    print("Reference:", referencia)
    print("Generated:", gerado)
    print("BLEU:", round(bleu, 3))
    print("ROUGE:", {k: round(v["f"], 3) for k, v in rouge_score.items()})
    print("BERTScore F1:", round(bert_f1, 3))
    print("-" * 60)

# Calcular médias
media_bleu = sum(bleu_scores) / len(bleu_scores)
media_rouge1 = sum(rouge_1_scores) / len(rouge_1_scores)
media_rouge2 = sum(rouge_2_scores) / len(rouge_2_scores)
media_rougel = sum(rouge_l_scores) / len(rouge_l_scores)
media_bert = sum(bert_f1_scores) / len(bert_f1_scores)

print("Médias dos resultados:")
print("Average BLEU:", round(media_bleu, 3))
print("Average ROUGE-1 F:", round(media_rouge1, 3))
print("Average ROUGE-2 F:", round(media_rouge2, 3))
print("Average ROUGE-L F:", round(media_rougel, 3))
print("Average BERTScore F1:", round(media_bert, 3))
