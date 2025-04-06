from transformers import MT5Tokenizer, MT5ForConditionalGeneration

model = MT5ForConditionalGeneration.from_pretrained("./modelo-final")
tokenizer = MT5Tokenizer.from_pretrained("./modelo-final")


def gerar_descricao(labels):
    input_text = "descreve: " + ", ".join(labels)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    output_ids = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# Exemplo de uso
exemplo = ["carro", "zona residencial", "semáforo"]
print("Input:", exemplo)
print("Descrição:", gerar_descricao(exemplo))
