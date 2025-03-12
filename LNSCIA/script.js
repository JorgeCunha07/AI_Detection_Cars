document.addEventListener('DOMContentLoaded', () => {
    console.log("Script carregado"); // Log para debug
    
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    if (!chatContainer || !userInput || !sendButton) {
        console.error("Elementos não encontrados:"); // Log para debug
        console.error("chatContainer:", chatContainer);
        console.error("userInput:", userInput);
        console.error("sendButton:", sendButton);
        return;
    }

    // Função para adicionar mensagem ao chat
    function addMessage(text, isUser) {
        console.log("Adicionando mensagem:", text); // Log para debug
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
        messageDiv.textContent = text;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Função para mostrar indicador de carregamento
    function showLoading() {
        const loadingDiv = document.createElement('div');
        loadingDiv.classList.add('message', 'bot-message', 'loading');
        loadingDiv.id = 'loading-indicator';
        loadingDiv.textContent = 'Llama 3 está pensando...';
        chatContainer.appendChild(loadingDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Função para remover indicador de carregamento
    function hideLoading() {
        const loadingDiv = document.getElementById('loading-indicator');
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }

    // Função para enviar mensagem ao backend
    async function sendMessage(message) {
        try {
            showLoading();
            const response = await fetch('http://localhost:5000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });

            if (!response.ok) {
                throw new Error('Erro na comunicação com o servidor');
            }

            const data = await response.json();
            hideLoading();
            addMessage(data.response, false);
        } catch (error) {
            hideLoading();
            addMessage('Erro: ' + error.message, false);
            console.error('Erro:', error);
        }
    }

    // Event listener para o botão de enviar
    sendButton.addEventListener('click', () => {
        console.log("Botão clicado"); // Log para debug
        const message = userInput.value.trim();
        if (message) {
            addMessage(message, true);
            userInput.value = '';
            sendMessage(message);
        }
    });

    // Event listener para a tecla Enter
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            console.log("Enter pressionado"); // Log para debug
            e.preventDefault();
            sendButton.click();
        }
    });

    // Mensagem inicial
    addMessage('Olá! Como posso ajudar você hoje?', false);
    console.log("Inicialização completa"); // Log para debug
}); 