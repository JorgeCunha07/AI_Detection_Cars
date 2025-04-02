from flask import Flask, request, jsonify
from flask_cors import CORS
from chat_interativo_transformer import generate_response, load_quiz_data, load_codigo_data, similar_text
import random

app = Flask(__name__)
CORS(app)  # Permite requisições de qualquer origem

# Estado global para armazenar o modo atual e dados
quiz_data = load_quiz_data()
codigo_data = load_codigo_data()
modo_quiz = False
modo_conversa = False
pergunta_atual = None
perguntas_usadas = set()  # Conjunto para rastrear perguntas já usadas

@app.route('/chat', methods=['POST'])
def chat():
    global modo_quiz, modo_conversa, pergunta_atual, perguntas_usadas
    
    try:
        data = request.json
        message = data.get('message', '').strip()
        history = data.get('history', [])
        
        if not message:
            return jsonify({'error': 'Mensagem não fornecida'}), 400
        
        # Processa comandos especiais
        if message.lower() == "quiz":
            modo_quiz = True
            modo_conversa = False
            perguntas_usadas.clear()  # Limpa o histórico de perguntas usadas
            pergunta_atual = random.choice(quiz_data) if quiz_data else None
            if pergunta_atual:
                response = f"[QUIZ] {pergunta_atual['pergunta']}\n(Digite 'dica' para ver a resposta correta e continuar respondendo)\n(Digite 'encerrar quiz' para voltar ao modo normal)"
            else:
                response = "Erro: Não foi possível carregar perguntas para o quiz."
                modo_quiz = False
            return jsonify({'response': response})
            
        elif message.lower() == "ajuda":
            response = """Comandos disponíveis:
- quiz: Entra no modo de testes com perguntas aleatórias
- artigo X: Consulta o Artigo X° do Código da Estrada
- conversa: Inicia uma conversa normal
- encerrar quiz: Sai do modo quiz
- dica: Mostra a resposta correta durante o quiz
- ajuda: Mostra esta mensagem
- sair: Encerra a conversa"""
            return jsonify({'response': response})
            
        elif message.lower() == "conversa":
            modo_quiz = False
            modo_conversa = True
            pergunta_atual = None
            perguntas_usadas.clear()  # Limpa o histórico de perguntas usadas
            response = "Olá! Sou o seu assistente especializado em condução e Código da Estrada. Estou aqui para ajudar com qualquer dúvida que tenha sobre condução, regras de trânsito, ou qualquer outro assunto relacionado. Como posso ajudar?"
            return jsonify({'response': response})
            
        elif message.lower() == "encerrar quiz":
            if modo_quiz:
                modo_quiz = False
                modo_conversa = True
                pergunta_atual = None
                perguntas_usadas.clear()  # Limpa o histórico de perguntas usadas
                response = "Saindo do modo QUIZ. Voltando ao modo de conversa normal."
            else:
                response = "Você não está no modo QUIZ atualmente."
            return jsonify({'response': response})
            
        elif message.lower().startswith("artigo"):
            try:
                artigo_num = message.split("artigo")[1].strip()
                for artigo in codigo_data.get("articles", []):
                    if artigo_num in artigo.get("article", "").lower():
                        response = f"Artigo {artigo['article']}\n{artigo['text']}\n\nReferência: {artigo['reference']}"
                        return jsonify({'response': response})
                response = "Artigo não encontrado. Tente especificar melhor (ex: 'artigo 1º')"
            except:
                response = "Formato inválido. Use 'artigo X°' ou 'artigo X'."
            return jsonify({'response': response})
            
        # Verifica se está no modo quiz e se a mensagem é 's' para continuar
        elif modo_quiz and message.lower() == "s":
            # Gera uma nova pergunta aleatória que ainda não foi usada
            perguntas_disponiveis = [q for q in quiz_data if q not in perguntas_usadas]
            if not perguntas_disponiveis:
                # Se todas as perguntas foram usadas, limpa o histórico e começa de novo
                perguntas_usadas.clear()
                perguntas_disponiveis = quiz_data
            
            pergunta_atual = random.choice(perguntas_disponiveis)
            perguntas_usadas.add(pergunta_atual)
            
            if pergunta_atual:
                response = f"[QUIZ] {pergunta_atual['pergunta']}\n(Digite 'dica' para ver a resposta correta e continuar respondendo)\n(Digite 'encerrar quiz' para voltar ao modo normal)"
            else:
                response = "Erro: Não foi possível carregar uma nova pergunta."
                modo_quiz = False
            return jsonify({'response': response})
            
        # Verifica se está no modo quiz e se há uma pergunta atual
        elif modo_quiz and pergunta_atual:
            if message.lower() == "dica":
                response = f"Dica: A resposta correta é:\n{pergunta_atual['respostas_corretas'][0]}"
                if "referencia" in pergunta_atual:
                    response += f"\nReferência: {pergunta_atual['referencia']}"
                return jsonify({'response': response})
            
            # Verificar resposta no modo quiz
            resposta_usuario = message.lower()
            correto = False
            
            for resposta_correta in pergunta_atual.get("respostas_corretas", []):
                if similar_text(resposta_usuario, resposta_correta.lower()):
                    correto = True
                    break
            
            if correto:
                response = f"✓ Correto!\nResposta: {pergunta_atual['respostas_corretas'][0]}"
            else:
                response = f"✗ Incorreto.\nResposta correta: {pergunta_atual['respostas_corretas'][0]}"
            
            if "referencia" in pergunta_atual:
                response += f"\nReferência: {pergunta_atual['referencia']}"
            
            response += "\n\nContinuar com o quiz? (s/n)"
            return jsonify({'response': response})
        
        # Modo de conversa normal
        historico_conversa = []
        
        # Adiciona instruções específicas para português de Portugal
        historico_conversa.append("Instruções: Responda sempre em português de Portugal, usando termos e expressões comuns em Portugal. Mantenha um tom profissional mas amigável.")
        
        # Adiciona o histórico da conversa
        for msg in history:
            if msg['role'] == 'user':
                historico_conversa.append(f"Utilizador: {msg['content']}")
            else:
                historico_conversa.append(f"Assistente: {msg['content']}")
        
        # Adiciona a mensagem atual
        historico_conversa.append(f"Utilizador: {message}")
        
        # Gera a resposta com o histórico completo
        response = generate_response(message, historico_conversa)
        
        # Garante que a resposta está em português de Portugal
        if not any(palavra in response.lower() for palavra in ['portugal', 'português', 'portuguesa', 'condutor', 'condução', 'estrada', 'trânsito', 'carro', 'moto', 'velocidade', 'multa', 'sinal']):
            response = "Peço desculpa, mas vou reformular a minha resposta em português de Portugal. " + response
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 