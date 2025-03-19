from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

@app.route('/')
def index():
    return send_file('index.html')

# Rota para servir o script.js que está no mesmo diretório
@app.route('/script.js')
def script_js():
    return send_file('script.js')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Mensagem vazia. Por favor, envie uma mensagem válida.'}), 400
    
    try:
        # Usando o executável local do Llama3 para gerar a resposta
        cmd = ["ollama", "run", "llama3", user_message]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Processa a saída do Llama3
        llama_response = result.stdout.strip()
        
        if not llama_response and result.stderr:
            return jsonify({'response': f'Erro ao executar Llama3: {result.stderr}'}), 500
        
        return jsonify({'response': llama_response})
    
    except Exception as e:
        return jsonify({'response': f'Erro ao processar a solicitação: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
