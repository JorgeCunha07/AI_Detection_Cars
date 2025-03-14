// script.js

// Seleciona os elementos do DOM
const sendButton = document.getElementById('send-button');
const userInput = document.getElementById('user-input');
const chatContainer = document.getElementById('chat-container');

// Função que envia a mensagem
async function sendMessage() {
  // Captura o valor atual e remove espaços extras
  const message = userInput.value.trim();
  if (!message) {
    alert('Por favor, insira uma mensagem válida.');
    return;
  }
  
  // Limpa o campo de mensagem imediatamente
  userInput.value = '';

  // Exibe a mensagem do usuário na interface
  const userMessageDiv = document.createElement('div');
  userMessageDiv.textContent = 'Você: ' + message;
  userMessageDiv.classList.add('message', 'user-message');
  chatContainer.appendChild(userMessageDiv);

  // Exibe um indicador de loading
  const loadingDiv = document.createElement('div');
  loadingDiv.textContent = 'Carregando...';
  loadingDiv.classList.add('message', 'loading');
  chatContainer.appendChild(loadingDiv);

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message: message })
    });
    const data = await response.json();
    
    // Remove o indicador de loading
    chatContainer.removeChild(loadingDiv);
    
    // Exibe a resposta do bot
    const botMessageDiv = document.createElement('div');
    botMessageDiv.textContent = 'Bot: ' + data.response;
    botMessageDiv.classList.add('message', 'bot-message');
    chatContainer.appendChild(botMessageDiv);
  } catch (error) {
    console.error('Erro ao enviar a mensagem:', error);
    chatContainer.removeChild(loadingDiv);
    const errorDiv = document.createElement('div');
    errorDiv.textContent = 'Erro ao comunicar com o servidor.';
    errorDiv.classList.add('message');
    chatContainer.appendChild(errorDiv);
  }
}

// Envia a mensagem ao clicar no botão
sendButton.addEventListener('click', async (event) => {
  event.preventDefault();
  sendMessage();
});

// Envia a mensagem ao pressionar Enter (sem Shift) no campo de texto
userInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});
