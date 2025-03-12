from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import subprocess
import json
import os

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Mensagem vazia. Por favor, envie uma mensagem válida.'}), 400
    
    try:
        # Usando Ollama para executar o modelo Llama 3
        cmd = ["ollama", "run", "llama3", user_message]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Processa a saída do Ollama
        llama_response = result.stdout.strip()
        
        # Se houver algum erro, verificamos a saída de erro
        if not llama_response and result.stderr:
            return jsonify({'response': f'Erro do Ollama: {result.stderr}'}), 500
        
        return jsonify({'response': llama_response})
    
    except Exception as e:
        return jsonify({'response': f'Erro ao processar a solicitação: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 