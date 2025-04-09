import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import re
import nltk
from nltk.stem import SnowballStemmer
import unicodedata

nltk.download('punkt', quiet=True)

# Stemmer português
stemmer = SnowballStemmer("portuguese")

# Normalização completa do texto (acentos, pontuação, espaços)
def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

# Diretório do modelo fine-tuned
model_dir = Path(__file__).resolve().parent.parent / "chat" / "gpt2-chat-finetuned"

# Carregar modelo e tokenizador
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Lista de palavras-chave do domínio do Código da Estrada
palavras_chave = [
    # Veículos e condução
    "carro", "veículo", "automóvel", "motociclo", "ciclomotor", "mota", "bicicleta", "trotinete", "camioneta", "camião",
    "trator", "reboque", "ambulância", "autocarro", "táxi", "uber", "condutor", "condutora", "carta", "conduzir",
    "marchas", "manobrar", "marcha-atrás", "volante", "travagem", "embraiagem", "pedal", "ignição", "motor", "avaria",

    # Infrações e segurança
    "multa", "coima", "infração", "pontos", "penalização", "crime", "sanção", "contraordenação", "alcoolemia", "balão",
    "drogas", "teste", "fiscalização", "GNR", "PSP", "polícia", "radar", "limite", "velocidade", "excesso", "segurança",
    "cinto", "cadeirinha", "criança", "animais", "gato", "cão", "capacete", "perigo", "prudência", "colisão", "acidente",
    "airbag", "ABS", "iluminação", "máximos", "médios", "nevoeiro", "travões",

    # Estacionamento e circulação
    "estacionar", "parar", "berma", "passeio", "faixa", "via", "passadeira", "rotunda", "cruzamento", "entroncamento",
    "zona escolar", "segunda fila", "garagem", "acostamento", "bloqueado", "obstrução", "zona reservada",
    "local proibido", "parquímetro", "parqueamento",

    # Sinalização e regras
    "sinal", "sinalização", "semáforo", "stop", "prioridade", "ceda passagem", "linha contínua", "linha descontínua",
    "marcação", "triângulo", "pisca", "piscas", "seta", "placa", "limite de velocidade", "proibição", "obrigatoriedade",

    # Vias e ambiente rodoviário
    "autoestrada", "estrada nacional", "IC", "IP", "AE", "via rápida", "ciclovia", "vias urbanas", "vias rurais",
    "passagem de nível", "túnel", "ponte", "viaduto", "travessia", "peões", "trânsito", "engarrafamento", "condições meteorológicas",

    # Documentação e obrigações
    "inspeção", "documentos", "registo", "livrete", "seguro", "declaração amigável", "matrícula", "licença",
    "vistoria", "certificado", "emissão", "renovação",

    # Situações específicas
    "acidente", "emergência", "assistência", "guincho", "obra", "desvio", "sinalização temporária",
    "pneu furado", "combustível", "gasóleo", "gasolina", "carregamento", "carro elétrico", "bateria", "carregador",

    # Extras
    "carpool", "zonas 30", "vias partilhadas", "circuito", "escola de condução", "exame", "instrutor", "aluno",
    "manobra", "baliza", "curva", "curva apertada", "curva perigosa", "regras de prioridade"
]

# Pré-processar os stems das palavras-chave uma vez
stems_keywords = set(stemmer.stem(k) for k in palavras_chave)

# Verifica se a pergunta pertence ao domínio
def pergunta_valida(pergunta: str) -> bool:
    pergunta_limpa = normalizar_texto(pergunta)
    stems_pergunta = [stemmer.stem(p) for p in pergunta_limpa.split()]
    return any(stem in stems_keywords for stem in stems_pergunta)

# Limpeza pós-geração
def post_process_resposta(generated_text: str, prompt: str) -> str:
    resposta = generated_text.replace(prompt, "").strip()
    resposta = resposta.split(".")[0].strip()

    if not resposta:
        return "Desculpe, não consegui gerar uma resposta."
    elif len(resposta.split()) < 4:
        return "Precisa de reformular a pergunta."
    else:
        return resposta + "."

# Pipeline de geração de resposta
def gerar_resposta_chat(user_input: str) -> str:
    if not pergunta_valida(user_input):
        return "Sou um assistente especializado apenas em temas do Código da Estrada. Reformule a sua pergunta."

    prompt = f"Usuário: {user_input}\nAssistente: " 
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            num_beams=3,
            temperature=0.7,
            top_p=0.9,
            top_k=30,
            repetition_penalty=2.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return post_process_resposta(generated_text, prompt)

# CLI
if __name__ == "__main__":
    print("🧠 Modo de inferência ativado. Escreva uma pergunta para testar o modelo.")
    print("❌ Digite 'sair' para encerrar.")

    while True:
        user_input = input("❓ Pergunta: ").strip()
        if user_input.lower() in ["sair", "exit", "q", "quit"]:
            print("👋 Encerrando inferência. Até à próxima!")
            break

        resposta = gerar_resposta_chat(user_input)
        print(f"🤖 Resposta: {resposta}")
