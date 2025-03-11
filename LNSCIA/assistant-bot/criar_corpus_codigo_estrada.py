import json
import os
import re

def criar_corpus_codigo_estrada():
    """
    Cria um corpus baseado no Código da Estrada de Portugal para o chatbot.
    O corpus será salvo em formato JSON compatível com o chatbot.
    """
    print("Criando corpus do Código da Estrada de Portugal...")
    
    # Estrutura para armazenar o corpus
    corpus = {
        "perguntas_respostas": []
    }
    
    # Adicionar regras básicas do Código da Estrada
    adicionar_regras_basicas(corpus)
    
    # Adicionar regras sobre sinais de trânsito
    adicionar_regras_sinais(corpus)
    
    # Adicionar regras sobre contraordenações
    adicionar_regras_contraordenacoes(corpus)
    
    # Adicionar regras sobre cartas de condução
    adicionar_regras_cartas_conducao(corpus)
    
    # Salvar o corpus em um arquivo JSON
    with open("corpus_codigo_estrada.json", "w", encoding="utf-8") as arquivo:
        json.dump(corpus, arquivo, ensure_ascii=False, indent=4)
    
    print(f"Corpus criado com sucesso! Total de {len(corpus['perguntas_respostas'])} regras adicionadas.")
    print("Arquivo salvo como: corpus_codigo_estrada.json")
    
    return corpus

def adicionar_regras_basicas(corpus):
    """Adiciona regras básicas do Código da Estrada."""
    regras = [
        {
            "padrao": r"(?i).*(?:velocidade máxima|limite de velocidade).*(?:zona urbana|localidade|cidade).*",
            "respostas": [
                "Em zonas urbanas, o limite de velocidade é geralmente de 50 km/h, salvo sinalização em contrário.",
                "Dentro das localidades, os veículos não podem circular a mais de 50 km/h, exceto quando a sinalização estabelecer outros limites.",
                "O Código da Estrada estabelece que em zonas urbanas o limite é de 50 km/h, a menos que exista sinalização específica."
            ]
        },
        {
            "padrao": r"(?i).*(?:velocidade máxima|limite de velocidade).*(?:autoestrada|auto-estrada|auto estrada).*",
            "respostas": [
                "Em autoestradas, o limite de velocidade é de 120 km/h para veículos ligeiros de passageiros e motociclos.",
                "Para veículos ligeiros de passageiros e motociclos, o limite em autoestradas é de 120 km/h. Veículos pesados têm limites inferiores.",
                "O Código da Estrada estabelece que em autoestradas o limite para veículos ligeiros é de 120 km/h."
            ]
        },
        {
            "padrao": r"(?i).*(?:velocidade máxima|limite de velocidade).*(?:via reservada|via rápida).*",
            "respostas": [
                "Em vias reservadas a automóveis e motociclos (vias rápidas), o limite é de 100 km/h para veículos ligeiros.",
                "Nas vias rápidas, os veículos ligeiros não podem exceder os 100 km/h, salvo sinalização em contrário.",
                "O Código da Estrada estabelece que em vias reservadas a automóveis e motociclos o limite é de 100 km/h para veículos ligeiros."
            ]
        },
        {
            "padrao": r"(?i).*(?:velocidade máxima|limite de velocidade).*(?:estrada nacional|estradas nacionais).*",
            "respostas": [
                "Nas estradas nacionais, o limite de velocidade é de 90 km/h para veículos ligeiros de passageiros e motociclos.",
                "Para veículos ligeiros, o limite em estradas nacionais é de 90 km/h, salvo sinalização em contrário.",
                "O Código da Estrada estabelece que em estradas nacionais o limite para veículos ligeiros é de 90 km/h."
            ]
        },
        {
            "padrao": r"(?i).*(?:taxa|nível|limite).*(?:álcool|alcool|alcoolemia).*(?:sangue|permitid|legal).*",
            "respostas": [
                "A taxa de álcool no sangue permitida para condutores normais é de 0,5 g/l. Para condutores em regime probatório, profissionais e de veículos de socorro, o limite é de 0,2 g/l.",
                "O limite legal de álcool no sangue é de 0,5 gramas por litro. Para condutores recém-encartados (menos de 3 anos) e profissionais, o limite é de 0,2 g/l.",
                "Segundo o Código da Estrada, a taxa máxima de alcoolemia permitida é de 0,5 g/l para a maioria dos condutores e 0,2 g/l para condutores em regime probatório ou profissionais."
            ]
        },
        {
            "padrao": r"(?i).*(?:distância|distancia).*(?:segurança|segura).*(?:veículo|veiculo|carro).*(?:frente|anterior).*",
            "respostas": [
                "O condutor deve manter uma distância de segurança que permita parar o veículo em caso de travagem brusca do veículo da frente. A regra dos 2 segundos é uma boa referência.",
                "A distância de segurança deve ser tal que permita imobilizar o veículo no espaço livre visível à sua frente. Uma forma prática é usar a regra dos 2 segundos.",
                "O Código da Estrada não especifica uma distância exata, mas exige que seja suficiente para evitar acidentes em caso de redução brusca de velocidade ou imobilização do veículo da frente."
            ]
        },
        {
            "padrao": r"(?i).*(?:prioridade|ceder passagem).*(?:rotunda|rotundas).*",
            "respostas": [
                "Numa rotunda, a prioridade é de quem circula dentro da rotunda. Quem vai entrar deve ceder passagem aos veículos que já se encontram na rotunda.",
                "Os veículos que pretendem entrar numa rotunda devem ceder passagem aos veículos que nela circulam, salvo sinalização em contrário.",
                "Segundo o Código da Estrada, quem circula dentro da rotunda tem prioridade sobre quem pretende entrar."
            ]
        },
        {
            "padrao": r"(?i).*(?:usar|utilizar|uso).*(?:telemóvel|celular|telefone).*(?:conduzir|dirigir|volante).*",
            "respostas": [
                "É proibido usar o telemóvel durante a condução, exceto se utilizar sistemas mãos-livres ou auricular que não impliquem manuseamento continuado.",
                "O Código da Estrada proíbe o uso de telemóvel durante a condução, a menos que seja com auricular ou sistema de mãos-livres que não exija manuseamento continuado.",
                "É expressamente proibido manusear o telemóvel durante a condução. O uso só é permitido com sistemas mãos-livres."
            ]
        },
        {
            "padrao": r"(?i).*(?:cinto de segurança|cinto).*(?:obrigatório|obrigatorio|usar|utilizar).*",
            "respostas": [
                "O uso do cinto de segurança é obrigatório para todos os ocupantes, tanto nos bancos da frente como nos traseiros.",
                "Todos os ocupantes de um veículo são obrigados a usar o cinto de segurança, com algumas exceções previstas na lei (como razões médicas certificadas).",
                "O Código da Estrada estabelece a obrigatoriedade do uso do cinto de segurança por todos os ocupantes do veículo."
            ]
        },
        {
            "padrao": r"(?i).*(?:criança|criancas|menor).*(?:cadeira|cadeirinha|sistema de retenção|sistema retenção).*",
            "respostas": [
                "As crianças com menos de 12 anos e altura inferior a 135 cm devem utilizar sistemas de retenção adequados ao seu tamanho e peso.",
                "É obrigatório o uso de cadeiras ou sistemas de retenção apropriados para crianças com menos de 12 anos e altura inferior a 135 cm.",
                "O Código da Estrada exige que crianças com menos de 12 anos e altura inferior a 135 cm utilizem sistemas de retenção homologados e adaptados ao seu peso e altura."
            ]
        }
    ]
    
    # Adicionar as regras ao corpus
    corpus["perguntas_respostas"].extend(regras)

def adicionar_regras_sinais(corpus):
    """Adiciona regras sobre sinais de trânsito."""
    regras = [
        {
            "padrao": r"(?i).*(?:sinal|sinais).*(?:vermelho|encarnado).*(?:semáforo|semaforo).*",
            "respostas": [
                "O sinal vermelho do semáforo indica proibição de avançar. Os condutores devem parar antes da linha de paragem ou, se não existir, antes do semáforo.",
                "Perante um sinal vermelho, é obrigatório parar e aguardar que o sinal mude para verde.",
                "O sinal vermelho do semáforo impõe a paragem obrigatória antes da linha de paragem ou do próprio semáforo."
            ]
        },
        {
            "padrao": r"(?i).*(?:sinal|sinais).*(?:amarelo|ambar|âmbar).*(?:semáforo|semaforo).*",
            "respostas": [
                "O sinal amarelo do semáforo indica que os condutores devem parar, a não ser que se encontrem tão perto que não o possam fazer em segurança.",
                "Perante um sinal amarelo fixo, deve parar, exceto se já estiver tão próximo que a paragem possa causar perigo.",
                "O sinal amarelo do semáforo significa que deve parar, a menos que não o possa fazer em condições de segurança."
            ]
        },
        {
            "padrao": r"(?i).*(?:sinal|sinais).*(?:verde).*(?:semáforo|semaforo).*",
            "respostas": [
                "O sinal verde do semáforo permite avançar, respeitando as regras de cedência de passagem.",
                "Perante um sinal verde, pode avançar, mas deve ceder passagem a peões ou veículos que ainda estejam a atravessar.",
                "O sinal verde autoriza os condutores a avançar, respeitando as regras gerais de trânsito."
            ]
        },
        {
            "padrao": r"(?i).*(?:sinal|sinais).*(?:stop|paragem obrigatória).*",
            "respostas": [
                "O sinal de STOP (paragem obrigatória) obriga o condutor a parar completamente o veículo antes de avançar, cedendo passagem a todos os veículos que circulem na via.",
                "Perante um sinal de STOP, é obrigatório parar completamente o veículo e só avançar quando for seguro, cedendo passagem aos veículos que circulem na via.",
                "O sinal de paragem obrigatória (STOP) exige uma paragem completa e a cedência de passagem a todos os veículos que circulem na via em que vai entrar."
            ]
        },
        {
            "padrao": r"(?i).*(?:sinal|sinais).*(?:cedência de passagem|ceder passagem).*",
            "respostas": [
                "O sinal de cedência de passagem obriga o condutor a ceder passagem a todos os veículos que circulem na via em que vai entrar, devendo parar se necessário.",
                "Perante um sinal de cedência de passagem, deve abrandar e, se necessário, parar para ceder passagem aos veículos que circulem na via em que vai entrar.",
                "O sinal triangular de cedência de passagem indica que deve ceder passagem aos veículos que circulem na via em que vai entrar ou que se aproximem pela direita."
            ]
        },
        {
            "padrao": r"(?i).*(?:linha|marcas).*(?:contínua|continua).*(?:ultrapassar|atravessar).*",
            "respostas": [
                "É proibido pisar ou transpor uma linha contínua, exceto para aceder a propriedades ou contornar obstáculos na via.",
                "A linha contínua não pode ser pisada ou transposta, salvo para acesso a propriedades ou para contornar um obstáculo na via.",
                "O Código da Estrada proíbe pisar ou transpor a linha contínua, com exceções para acesso a propriedades ou para contornar obstáculos."
            ]
        },
        {
            "padrao": r"(?i).*(?:linha|marcas).*(?:descontínua|descontinua|tracejada).*(?:ultrapassar|atravessar).*",
            "respostas": [
                "A linha descontínua (tracejada) pode ser pisada ou transposta durante manobras como ultrapassagens, desde que sejam respeitadas as regras de segurança.",
                "É permitido pisar ou transpor uma linha descontínua para realizar manobras como ultrapassagens, respeitando as regras de segurança.",
                "A linha descontínua (tracejada) indica que é permitido pisar ou transpor a linha para realizar manobras, respeitando as regras de segurança."
            ]
        }
    ]
    
    # Adicionar as regras ao corpus
    corpus["perguntas_respostas"].extend(regras)

def adicionar_regras_contraordenacoes(corpus):
    """Adiciona regras sobre contraordenações."""
    regras = [
        {
            "padrao": r"(?i).*(?:contraordenação|contraordenacao|multa).*(?:leve|ligeira).*",
            "respostas": [
                "As contraordenações leves são puníveis com coima de €30 a €150 para condutores de veículos ligeiros.",
                "Uma contraordenação leve implica uma coima entre €30 e €150 para condutores de veículos ligeiros.",
                "O Código da Estrada estabelece que as contraordenações leves são sancionadas com coimas de €30 a €150 para condutores de veículos ligeiros."
            ]
        },
        {
            "padrao": r"(?i).*(?:contraordenação|contraordenacao|multa).*(?:grave).*",
            "respostas": [
                "As contraordenações graves são puníveis com coima de €120 a €600 para condutores de veículos ligeiros e podem incluir sanção acessória de inibição de conduzir.",
                "Uma contraordenação grave implica uma coima entre €120 e €600 para condutores de veículos ligeiros, podendo haver inibição de conduzir.",
                "O Código da Estrada estabelece que as contraordenações graves são sancionadas com coimas de €120 a €600 para condutores de veículos ligeiros, podendo incluir inibição de conduzir."
            ]
        },
        {
            "padrao": r"(?i).*(?:contraordenação|contraordenacao|multa).*(?:muito grave).*",
            "respostas": [
                "As contraordenações muito graves são puníveis com coima de €250 a €1250 para condutores de veículos ligeiros e incluem sanção acessória de inibição de conduzir.",
                "Uma contraordenação muito grave implica uma coima entre €250 e €1250 para condutores de veículos ligeiros, incluindo inibição de conduzir.",
                "O Código da Estrada estabelece que as contraordenações muito graves são sancionadas com coimas de €250 a €1250 para condutores de veículos ligeiros, incluindo inibição de conduzir."
            ]
        },
        {
            "padrao": r"(?i).*(?:excesso de velocidade).*(?:contraordenação|contraordenacao|multa).*",
            "respostas": [
                "O excesso de velocidade pode ser contraordenação leve (até 20 km/h), grave (de 20 a 40 km/h) ou muito grave (mais de 40 km/h), com coimas e sanções proporcionais.",
                "Exceder o limite de velocidade até 20 km/h é contraordenação leve, entre 20 e 40 km/h é grave, e mais de 40 km/h é muito grave, com coimas e sanções correspondentes.",
                "O Código da Estrada classifica o excesso de velocidade como contraordenação leve, grave ou muito grave, dependendo de quanto se excede o limite, com coimas e sanções proporcionais."
            ]
        },
        {
            "padrao": r"(?i).*(?:álcool|alcool|alcoolemia).*(?:contraordenação|contraordenacao|multa).*",
            "respostas": [
                "Conduzir com taxa de álcool entre 0,5 e 0,8 g/l é contraordenação grave, entre 0,8 e 1,2 g/l é muito grave, e acima de 1,2 g/l é crime.",
                "Taxa de álcool entre 0,5 e 0,8 g/l constitui contraordenação grave, entre 0,8 e 1,2 g/l é muito grave, e acima de 1,2 g/l configura crime de condução sob influência de álcool.",
                "O Código da Estrada estabelece que conduzir com taxa de álcool entre 0,5 e 1,2 g/l constitui contraordenação (grave ou muito grave), e acima de 1,2 g/l é crime."
            ]
        },
        {
            "padrao": r"(?i).*(?:telemóvel|celular|telefone).*(?:contraordenação|contraordenacao|multa).*",
            "respostas": [
                "O uso do telemóvel durante a condução constitui contraordenação grave, punível com coima de €120 a €600 e sanção acessória de inibição de conduzir.",
                "Usar o telemóvel enquanto conduz é uma contraordenação grave, com coima entre €120 e €600 e possível inibição de conduzir.",
                "O Código da Estrada classifica o uso do telemóvel durante a condução como contraordenação grave, com coimas de €120 a €600 e sanção acessória."
            ]
        }
    ]
    
    # Adicionar as regras ao corpus
    corpus["perguntas_respostas"].extend(regras)

def adicionar_regras_cartas_conducao(corpus):
    """Adiciona regras sobre cartas de condução."""
    regras = [
        {
            "padrao": r"(?i).*(?:idade mínima|idade minima).*(?:carta de condução|carta conducao).*(?:categoria|cat) (?:A|B).*",
            "respostas": [
                "A idade mínima para obter carta de condução da categoria B (automóveis ligeiros) é 18 anos. Para a categoria A (motociclos), a idade mínima varia conforme a potência.",
                "Para obter carta de categoria B é necessário ter pelo menos 18 anos. Para categoria A, depende da subcategoria (A1, A2 ou A).",
                "O Código da Estrada estabelece 18 anos como idade mínima para carta de categoria B. Para categoria A, varia entre 16 anos (A1), 18 anos (A2) e 24 anos (A)."
            ]
        },
        {
            "padrao": r"(?i).*(?:carta de condução|carta conducao).*(?:categoria|cat) A.*(?:o que|qual|quais|permite).*",
            "respostas": [
                "A categoria A permite conduzir motociclos e triciclos. Existem subcategorias: A1 (até 125cc), A2 (até 35kW) e A (sem restrição de potência).",
                "A carta de categoria A habilita a conduzir motociclos e triciclos, com subcategorias conforme a potência e cilindrada.",
                "A categoria A da carta de condução permite conduzir motociclos e triciclos, com diferentes subcategorias dependendo da potência."
            ]
        },
        {
            "padrao": r"(?i).*(?:carta de condução|carta conducao).*(?:categoria|cat) B.*(?:o que|qual|quais|permite).*",
            "respostas": [
                "A categoria B permite conduzir automóveis ligeiros com peso máximo até 3500kg e até 8 lugares além do condutor.",
                "A carta de categoria B habilita a conduzir automóveis ligeiros de passageiros e mercadorias até 3500kg e máximo de 9 lugares (incluindo o condutor).",
                "A categoria B da carta de condução permite conduzir veículos ligeiros com massa máxima de 3500kg e até 8 passageiros além do condutor."
            ]
        },
        {
            "padrao": r"(?i).*(?:regime probatório|periodo probatorio|período probatório).*(?:carta de condução|carta conducao).*",
            "respostas": [
                "Os titulares de carta de condução ficam em regime probatório durante os primeiros 3 anos após a obtenção da primeira carta, com regras mais rigorosas.",
                "O regime probatório aplica-se durante os primeiros 3 anos após obter a primeira carta de condução, com limites mais baixos de álcool (0,2 g/l) e outras restrições.",
                "Durante os primeiros 3 anos de carta, o condutor está em regime probatório, com taxa de álcool máxima de 0,2 g/l e cassação da carta se cometer contraordenação muito grave ou duas graves."
            ]
        },
        {
            "padrao": r"(?i).*(?:renovar|renovação).*(?:carta de condução|carta conducao).*",
            "respostas": [
                "A carta de condução deve ser renovada aos 30, 40, 50, 60, 65 e 70 anos, e depois de 2 em 2 anos. Condutores profissionais têm regras específicas.",
                "A renovação da carta de condução é obrigatória aos 30, 40, 50, 60, 65 e 70 anos, e posteriormente de 2 em 2 anos, mediante exame médico.",
                "O Código da Estrada estabelece que a carta deve ser renovada aos 30, 40, 50, 60, 65 e 70 anos, e depois a cada 2 anos, com apresentação de atestado médico."
            ]
        },
        {
            "padrao": r"(?i).*(?:pontos|sistema de pontos).*(?:carta de condução|carta conducao).*",
            "respostas": [
                "No sistema de pontos, cada condutor inicia com 12 pontos. Contraordenações graves deduzem 2 pontos, muito graves deduzem 4 pontos, e crimes rodoviários deduzem 6 pontos.",
                "O sistema de pontos atribui 12 pontos iniciais, que são reduzidos conforme as infrações cometidas: -2 para graves, -4 para muito graves e -6 para crimes.",
                "Cada condutor tem 12 pontos na carta. Perde pontos por infrações: 2 por contraordenação grave, 4 por muito grave e 6 por crime rodoviário. Com 0 pontos, a carta é cassada."
            ]
        }
    ]
    
    # Adicionar as regras ao corpus
    corpus["perguntas_respostas"].extend(regras)

def integrar_corpus_ao_chatbot(corpus_file):
    """
    Integra o corpus do Código da Estrada ao conhecimento do chatbot.
    
    Args:
        corpus_file: Caminho para o arquivo JSON do corpus
    """
    # Verificar se o arquivo do corpus existe
    if not os.path.exists(corpus_file):
        print(f"Erro: O arquivo {corpus_file} não foi encontrado.")
        return False
    
    # Verificar se o arquivo de conhecimento do chatbot existe
    conhecimento_file = "conhecimento_chatbot.json"
    if not os.path.exists(conhecimento_file):
        print(f"Erro: O arquivo {conhecimento_file} não foi encontrado.")
        return False
    
    try:
        # Carregar o corpus
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        
        # Carregar o conhecimento atual do chatbot
        with open(conhecimento_file, 'r', encoding='utf-8') as f:
            conhecimento = json.load(f)
        
        # Adicionar as regras do corpus ao conhecimento do chatbot
        conhecimento["perguntas_respostas"].extend(corpus["perguntas_respostas"])
        
        # Salvar o conhecimento atualizado
        with open(conhecimento_file, 'w', encoding='utf-8') as f:
            json.dump(conhecimento, f, ensure_ascii=False, indent=4)
        
        print(f"Corpus do Código da Estrada integrado com sucesso ao chatbot!")
        print(f"Foram adicionadas {len(corpus['perguntas_respostas'])} novas regras ao conhecimento.")
        return True
    
    except Exception as e:
        print(f"Erro ao integrar o corpus: {str(e)}")
        return False

# Executar o script
if __name__ == "__main__":
    # Criar o corpus
    corpus = criar_corpus_codigo_estrada()
    
    # Perguntar se deseja integrar ao chatbot
    resposta = input("\nDeseja integrar este corpus ao conhecimento do chatbot? (s/n): ").strip().lower()
    if resposta == 's':
        integrar_corpus_ao_chatbot("corpus_codigo_estrada.json") 