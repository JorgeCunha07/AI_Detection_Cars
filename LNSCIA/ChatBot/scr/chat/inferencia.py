import re
import nltk
import torch
import unicodedata

from pathlib import Path
from nltk.stem import SnowballStemmer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEBUG_MODE = False  # <- mudar para False para desligar o debug

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

palavras_chave = [
    # Veículos e condução
    "carro", "veículo", "automóvel", "motociclo", "ciclomotor", "mota", "bicicleta", "bicicleta elétrica", "trotinete", 
    "trotinete elétrica", "camioneta", "camião", "trator", "reboque", "ambulância", "autocarro", "táxi", "uber", 
    "condutor", "condutora", "carta", "conduzir", "carta caducada", "volante", "direção", "ignição", "motor", "avaria",

    # Acessórios e comportamento de condução
    "auriculares", "auscultadores", "telemóvel", "alta-voz", "gps", "música", "distração", "mãos-livres", 
    "auricular", "telefone", "suporte homologado", "óculos de sol", "sem camisa", "ar condicionado",

    # Infrações e segurança
    "multa", "coima", "infração", "pontos", "penalização", "crime", "sanção", "contraordenação", "alcoolemia", 
    "balão", "drogas", "teste", "fiscalização", "GNR", "PSP", "polícia", "radar", "velocidade", "excesso", "segurança",
    "prudência", "colisão", "acidente", "perigo", "travões", "airbag", "cinto", "capacete", "colete refletor",

    # Estacionamento e circulação
    "estacionar", "parar", "berma", "passeio", "faixa", "faixa do meio", "via", "ciclovia", "rotunda", "cruzamento",
    "entroncamento", "zona escolar", "segunda fila", "garagem", "acostamento", "bloqueado", "obstrução",
    "zona reservada", "parquímetro", "sentido contrário", "marcha atrás", "curva", "curva apertada", "desvio", "obras",

    # Sinalização e regras
    "sinal", "sinalização", "sinal coberto", "semáforo", "amarelo a piscar", "stop", "prioridade", 
    "ceda passagem", "linha contínua", "linha descontínua", "triângulo", "pisca", "quatro piscas", "seta", "placa", 
    "limite de velocidade", "proibição", "obrigatoriedade", "lombas", "passagem de nível", "sinais temporários",

    # Vias e ambiente rodoviário
    "autoestrada", "estrada nacional", "IC", "IP", "AE", "via rápida", "vias urbanas", "vias rurais",
    "túnel", "ponte", "viaduto", "passadeira", "peões", "trânsito", "engarrafamento", "visibilidade", 
    "nevoeiro", "chuva", "condições meteorológicas", "derrapagem",

    # Documentação e obrigações
    "inspeção", "documentos", "registo", "livrete", "seguro", "declaração amigável", "matrícula", "licença",
    "vistoria", "certificado", "emissão", "renovação", "legalização", "carta estrangeira",

    # Transporte e ocupantes
    "criança", "cadeirinha", "banco da frente", "animais", "cão", "gato", "passageiro", "ocupante", 
    "autocarro escolar", "elétrico", "transportes públicos", "taxista",

    # Situações e manobras
    "ultrapassagem", "ultrapassar", "dar passagem", "ceder passagem", "entrar na rotunda", 
    "sair da rotunda", "sinalizar", "mudar de faixa", "descarregar", "embarcar", "desembarcar", "virar", 
    "desligar motor", "paragem", "emergência", "guincho", "luz fundida", "limpa para-brisas", "buzina", 
    "manobra perigosa", "abertura de portas", "ligar piscas",

    # Palavras extras
    "banco traseiro", "faixa de rodagem", "direção assistida", "condutor isento", "exame de condução",
    "instrutor", "escola de condução", "zona de travagem", "rede de proteção", "estacionamento irregular",
    "curva sem visibilidade", "zona de ultrapassagem", "condutor profissional"
]

# Pré-processar os stems das palavras-chave uma vez
stems_keywords = set(stemmer.stem(k) for k in palavras_chave)

def debug_stems(pergunta: str):
    pergunta_limpa = normalizar_texto(pergunta)
    stems_pergunta = [stemmer.stem(p) for p in pergunta_limpa.split()]
    matches = [s for s in stems_pergunta if s in stems_keywords]
    
    print("❓ Pergunta:", pergunta)
    print("🔤 Stems da pergunta:", stems_pergunta)
    print("✅ Stems que coincidem com palavras-chave:", matches)
    
    if not matches:
        print("🚫 Nenhum stem corresponde às palavras-chave.")

# Verifica se a pergunta pertence ao domínio
def pergunta_valida(pergunta: str) -> bool:
    pergunta_limpa = normalizar_texto(pergunta)
    stems_pergunta = [stemmer.stem(p) for p in pergunta_limpa.split()]
    
    if DEBUG_MODE:
        debug_stems(pergunta)

    return any(stem in stems_keywords for stem in stems_pergunta)

# Limpeza pós-geração
def post_process_resposta(generated_text: str, prompt: str) -> str:
    resposta = generated_text.replace(prompt, "").strip()
    
    if DEBUG_MODE:
        print("🔎 Resposta gerada bruta:", resposta)

    # Em vez de só a primeira frase:
    if not resposta or len(resposta.split()) < 4:
        return "Precisa de reformular a pergunta."
    
    if not resposta.endswith("."):
        resposta += "."
    
    return resposta

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
            max_new_tokens=40,          # Limite de tokens gerados
            do_sample=True,             # Habilita amostragem
            num_beams=3,                # Número de feixes para busca
            temperature=0.7,            # Controle de aleatoriedade
            top_p=0.9,                  # Amostragem de núcleo
            top_k=30,                   # Limite de amostragem
            repetition_penalty=2.0,     # Penalização de repetição
            pad_token_id=tokenizer.eos_token_id,            # Token de preenchimento
            eos_token_id=tokenizer.eos_token_id,            # Token de fim de sequência
            no_repeat_ngram_size=3,     # Tamanho do n-grama a evitar repetição
            num_return_sequences=1,     # Número de sequências a retornar
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return post_process_resposta(generated_text, prompt)

# CLI para testar o modelo sem invocar o main
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