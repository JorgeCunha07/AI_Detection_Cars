from transformers import T5Tokenizer, MT5ForConditionalGeneration

# Carregar modelo e tokenizer treinado
model = MT5ForConditionalGeneration.from_pretrained("./modelo-final")
tokenizer = T5Tokenizer.from_pretrained("./modelo-final")


def gerar_descricao(labels):
    input_text = "Gere uma frase que descreva a seguinte cena: " + ", ".join(labels)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# Exemplo de uso
exemplo = ["carro", "zona residencial", "semáforo"]
print("Input:", exemplo)
print("Descrição:", gerar_descricao(exemplo))

# Exemplo de uso
exemplo = ["carro", "peão", "passadeira", "semáforo", "chuva", "amanhecer", "zona_residencial"]
print("Input:", exemplo)
print("Descrição:", gerar_descricao(exemplo))

# Exemplo de uso
exemplo = ["carro", "autocarro", "semáforo", "noite", "túnel"]
print("Input:", exemplo)
print("Descrição:", gerar_descricao(exemplo))

# Exemplo de uso
exemplo = ["peões", "ciclistas", "dia", "zona_residencial"]
print("Input:", exemplo)
print("Descrição:", gerar_descricao(exemplo))

# Exemplo de uso
exemplo = ["sinal_de_stop"]
print("Input:", exemplo)
print("Descrição:", gerar_descricao(exemplo))

# Exemplo de uso
exemplo = ["passadeira", "sinal_de_passadeira"]
print("Input:", exemplo)
print("Descrição:", gerar_descricao(exemplo))

# Exemplo de uso
exemplo = ["neve", "amanhecer", "zona_residencial"]
print("Input:", exemplo)
print("Descrição:", gerar_descricao(exemplo))

# Exemplo de uso
exemplo = ["ciclista", "sinal_de_limite_de_velocidade", "dia", "túnel"]
print("Input:", exemplo)
print("Descrição:", gerar_descricao(exemplo))
