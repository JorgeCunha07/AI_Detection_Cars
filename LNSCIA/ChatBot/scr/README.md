# ChatBot do Código da Estrada Português

Este projeto implementa um chatbot interativo para ajudar na aprendizagem e consulta do Código da Estrada Português, com modo de quiz e consulta de artigos.

## Novidades: Sistema de Similaridade baseado em Transformers

A nova versão do chatbot agora oferece uma implementação baseada em modelos de linguagem **Transformer** para analisar as respostas dos utilizadores, substituindo o sistema anterior baseado em regras manuais.

### Principais Melhorias:

1. **Compreensão Semântica Superior**
   - Entende o significado das frases, não apenas palavras individuais
   - Reconhece respostas corretas mesmo quando expressas de formas diferentes
   - Maior naturalidade na interação com o utilizador

2. **Suporte a Múltiplas Variações**
   - Interpreta automaticamente diferentes formas de expressar a mesma informação
   - Entende sinónimos e variações de formato sem regras manuais extensas

3. **Maior Precisão**
   - Melhor distinção entre respostas corretas e incorretas
   - Redução de falsos negativos (respostas corretas marcadas como erradas)
   - Melhor contexto para perguntas específicas do Código da Estrada

## Funcionalidades

- **Chat Interativo**: Conversação livre sobre o Código da Estrada
- **Modo Quiz**: Teste os seus conhecimentos com perguntas aleatórias
- **Consulta de Artigos**: Acesso direto a artigos específicos do Código
- **Similaridade Avançada**: Reconhecimento de respostas corretas com variações

## Requisitos

- Python 3.7 ou superior
- Bibliotecas listadas em `requirements.txt`
- Mínimo de 4GB de RAM (8GB recomendado)
- Espaço em disco: ~500MB (para modelos)
- GPU opcional (melhora significativamente o desempenho)

## Instalação

1. Clone o repositório
2. Execute o script de configuração:

```bash
cd ChatBot/scr3
python setup.py
```

O script de configuração irá:
- Verificar a versão do Python
- Instalar as dependências necessárias
- Verificar a presença dos arquivos de dados
- Pré-baixar o modelo transformer para uso offline

## Uso

### Iniciar o Chatbot com Sistema Transformer (recomendado)

```bash
python chat_interativo_transformer.py
```

### Iniciar o Chatbot com Sistema Baseado em Regras (fallback)

```bash
python chat_interativo_fixed.py
```

### Comandos Disponíveis no Chatbot

- `quiz` - Inicia o modo de perguntas e respostas
- `artigo X` - Consulta o Artigo X° do Código da Estrada
- `encerrar quiz` - Sai do modo quiz
- `ajuda` - Mostra a lista de comandos disponíveis
- `sair` - Encerra a conversa

## Exemplos de Respostas Reconhecidas

O sistema baseado em transformer consegue reconhecer as seguintes variações como corretas:

### Pergunta: Qual é a velocidade máxima permitida para automóveis ligeiros em autoestradas?
- "120 km/h"
- "120 quilómetros por hora"
- "cento e vinte quilómetros por hora"
- "É de 120 km por hora"
- "A velocidade máxima é 120"

### Pergunta: Qual a distância lateral mínima ao ultrapassar velocípedes?
- "1,5 metros"
- "Um metro e meio"
- "1,5m"
- "Um metro e cinquenta centímetros"
- "A distância mínima é de 1,5 metros"

## Configuração Avançada

É possível configurar o comportamento do sistema alterando o modelo transformer utilizado:

```python
chatbot = ChatBotCodigoEstrada(
    use_transformer=True,
    transformer_model="paraphrase-multilingual-mpnet-base-v2"  # Modelo alternativo
)
```

Modelos recomendados para português:
- `distiluse-base-multilingual-cased-v1` (padrão, mais rápido)
- `paraphrase-multilingual-mpnet-base-v2` (melhor qualidade, mais lento)
- `LaBSE` (bom para comparação entre diferentes estruturas de frase)

## Resolução de Problemas

### Erro ao carregar modelo transformer
Se o modelo não conseguir carregar, o sistema automaticamente utilizará o modo de fallback baseado em regras.

### Perguntas para o quiz não carregadas
Verifique se o arquivo `questions_dataset_enhanced.json` está presente no diretório.

### Modelo de chat não responde corretamente
Certifique-se de que o modelo foi treinado corretamente e que os arquivos do modelo estão presentes em `./trained_model`.

### Erro de memória
Os modelos transformer necessitam de memória significativa. Se estiver tendo problemas, considere:
- Fechar outras aplicações a consumir memória
- Mudar para um modelo mais leve (`distiluse-base-multilingual-cased-v1`)
- Usar o modo baseado em regras (`use_transformer=False`) 