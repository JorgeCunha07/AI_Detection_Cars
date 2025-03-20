import json
import random
import numpy as np

# ---------------------------
# Dados Sintéticos mais Ricos
# ---------------------------

# Lista base de labels (rótulos básicos)
possible_labels = [
    "peao", "pesoes", "camião", "camiões", "autocarro", "camioneta", "passadeira",
    "semaforo", "carro", "carros"
]

# Lista de sinais de perigo
danger_signs = [
    "Curva à direita", "Curva à esquerda", "Curva à direita e contracurva", "Curva à esquerda e contracurva",
    "Lomba", "Depressão", "Lomba ou depressão", "Descida perigosa", "Subida de inclinação acentuada", "Passagem estreita",
    "Pavimento escorregadio", "Projecção de gravilha", "Bermas baixas", "Saída num cais ou precipício", "Queda de pedras",
    "Ponte móvel", "Neve ou gelo", "Vento lateral", "Visibilidade insuficiente", "Crianças", "Idosos",
    "Passagem de peões", "Travessia de peões", "Saída de ciclistas", "Cavaleiros", "Animais", "Animais selvagens",
    "Túnel", "Pista de aviação", "Sinalização luminosa", "Trabalhos na via", "Cruzamento ou entroncamento",
    "Trânsito nos dois sentidos", "Passagem de nível com guarda", "Passagem de nível sem guarda",
    "Intersecção com via onde circulam veículos sobre carris", "Outros perigos", "Congestionamento",
    "Obstrução da via", "Local de passagem de nível sem guarda", "Local de passagem de nível sem guarda com duas ou mais vias"
]
danger_signs = list(dict.fromkeys(danger_signs))

# Lista de sinais de cedência de passagem
cedencia_signs = [
    "Cedência de passagem",
    "Paragem obrigatória no cruzamento ou entroncamento",
    "Via com prioridade",
    "Fim de via com prioridade",
    "Cedência de passagem nos estreitamentos da faixa de rodagem",
    "Prioridade nos estreitamentos da faixa de rodagem",
    "Aproximação de rotunda",
    "Cruzamento com via sem prioridade",
    "Entroncamento com via sem prioridade"
]
cedencia_signs = list(dict.fromkeys(cedencia_signs))

# Lista de sinais de proibição
proibicao_signs = [
    "Trânsito proibido",
    "Trânsito proibido a automóveis e motociclos com carro",
    "Trânsito proibido a automóveis pesados",
    "Trânsito proibido a automóveis de mercadorias",
    "Trânsito proibido a automóveis de mercadorias de peso total superior a ...t",
    "Trânsito proibido a motociclos simples",
    "Trânsito proibido a ciclomotores",
    "Trânsito proibido a velocípedes",
    "Trânsito proibido a veículos agrícolas",
    "Trânsito proibido a veículos de tracção animal",
    "Trânsito proibido a carros de mão",
    "Trânsito proibido a peões",
    "Trânsito proibido a cavaleiros",
    "Trânsito proibido a veículos com reboque",
    "Trânsito proibido a veículos com reboque de dois ou mais eixos",
    "Trânsito proibido a veículos transportando mercadorias perigosas",
    "Trânsito proibido a veículos transportando produtos facilmente inflamáveis ou explosivos",
    "Trânsito proibido a veículos transportando produtos susceptíveis de poluírem as águas",
    "Trânsito proibido a automóveis e motociclos",
    "Trânsito proibido a automóveis de mercadorias e a veículos a motor com reboque",
    "Trânsito proibido a automóveis, a motociclos e a veículos de tracção animal",
    "Trânsito proibido a automóveis de mercadorias e a veículos de tracção animal",
    "Trânsito proibido a peões, a animais e a veículos que não sejam automóveis ou motociclos",
    "Trânsito proibido a veículos de duas rodas",
    "Trânsito proibido a veículos de peso por eixo superior a ...t",
    "Trânsito proibido a veículos de peso total superior a ...t",
    "Trânsito proibido a veículos ou conjunto de veículos de comprimento superior a ...m",
    "Trânsito proibido a veículos de largura superior a ...m",
    "Trânsito proibido a veículos de altura superior a ...m",
    "Proibição de transitar a menos de ...m do veículo precedente",
    "Proibição de virar à direita",
    "Proibição de virar à esquerda",
    "Proibição de inversão do sentido de marcha",
    "Proibição de exceder a velocidade máxima de ...Km/h",
    "Proibição de ultrapassar",
    "Proibição de ultrapassar para automóveis pesados",
    "Proibição de ultrapassar para motociclos e ciclomotores",
    "Estacionamento proibido",
    "Paragem e estacionamento proibidos",
    "Proibição de sinais sonoros",
    "Paragem obrigatória na alfândega",
    "Outras paragens obrigatórias",
    "Fim de todas as proibições impostas anteriormente por sinalização a veículos em marcha",
    "Fim da limitação de velocidade",
    "Fim da proibição de ultrapassar",
    "Fim da proibição de ultrapassar para automóveis pesados",
    "Fim da proibição de ultrapassar para motociclos e ciclomotores",
    "Fim de paragem ou estacionamento proibidos",
    "Fim da proibição de sinais sonoros"
]
proibicao_signs = list(dict.fromkeys(proibicao_signs))

# Lista de sinais de obrigação
obrigacao_signs = [
    "Sentido obrigatório",
    "Sentidos obrigatórios possíveis",
    "Obrigação de contornar a placa ou obstáculo",
    "Rotunda",
    "Via obrigatória para automóveis de mercadorias",
    "Via obrigatória para automóveis pesados",
    "Via reservada a veículos de transporte público",
    "Pista obrigatória para velocípedes",
    "Pista obrigatória para peões",
    "Pista obrigatória para cavaleiros",
    "Pista obrigatória para gado e manada",
    "Pista obrigatória para peões e velocípedes",
    "Obrigação de transitar à velocidade mínima de … km/h",
    "Obrigação de utilizar correntes de neve",
    "Obrigação de utilizar as luzes de cruzamento (médios) acesas",
    "Fim da via obrigatória para automóveis de mercadorias",
    "Fim da via obrigatória para automóveis pesados",
    "Fim da via reservada a veículos de transporte público",
    "Fim da pista obrigatória para velocípedes",
    "Fim da pista obrigatória para peões",
    "Fim da pista obrigatória para cavaleiros",
    "Fim da pista obrigatória para gado em manada",
    "Fim da pista obrigatória para peões e velocípedes",
    "Fim da obrigação de transitar à velocidade mínima de … km/h",
    "Fim da obrigação de utilizar correntes de neve",
    "Fim da obrigação de utilizar as luzes de cruzamento acesas"
]
obrigacao_signs = list(dict.fromkeys(obrigacao_signs))

# Lista de sinais de informação
informacao_signs = [
    "Estacionamento autorizado",
    "Hospital",
    "Trânsito de sentido único",
    "Via pública sem saída",
    "Correntes de neve recomendadas",
    "Velocidade recomendada",
    "Passagem para peões",
    "Passagem desnivelada para peões",
    "Hospital com urgência médica",
    "Posto de socorros",
    "Oficina",
    "Telefone",
    "Posto de abastecimento de combustível",
    "Posto de abastecimento de combustível com GPL",
    "Posto de abastecimento de combustível com serviço a veículos eléctricos",
    "Posto de abastecimento de combustível com GPL e com serviço a veículos eléctricos",
    "Parque de campismo",
    "Parque para reboques de campismo",
    "Parque misto para campismo e reboques de campismo",
    "Telefone de emergência",
    "Pousada ou estalagem",
    "Albergue",
    "Pousada de juventude",
    "Turismo rural",
    "Hotel",
    "Restaurante",
    "Café ou bar",
    "Paragem de veículos de transporte colectivo de passageiros",
    "Paragem de veículos de transporte colectivo de passageiros que transitem sobre carris",
    "Paragem de veículos afectos ao transporte de crianças",
    "Aeroporto",
    "Posto de informações",
    "Estação de radiodifusão",
    "Auto estrada",
    "Via reservada a automóveis e motociclos",
    "Escapatória",
    "Inversão do sentido de marcha",
    "Limites de velocidade",
    "Identificação de país",
    "Praticabilidade da via",
    "Número e sentido das vias de trânsito",
    "Supressão de via de trânsito",
    "Via verde",
    "Centro de inspecções",
    "Túnel",
    "Fim da recomendação do uso de correntes de neve",
    "Fim da velocidade recomendada",
    "Fim de auto",
    "Fim de via reservada a automóveis e motociclos",
    "Fim de estacionamento autorizado",
    "Fim de túnel",
    "Velocidade média",
    "Velocidade instantânea",
    "Lanço com cobrança electrónica de portagem",
    "Fim de lanço com cobrança electrónica de portagem"
]
informacao_signs = list(dict.fromkeys(informacao_signs))

# Combina todas as listas de rótulos
combined_labels = (possible_labels + danger_signs + cedencia_signs +
                   proibicao_signs + obrigacao_signs + informacao_signs)

# Listas fixas para atributos
weather_conditions = ["ensolarado", "nublado", "chuvoso", "tempestuoso", "com nevoeiro", "vento forte"]
time_of_day = ["de manhã cedo", "no fim da manhã", "à tarde", "ao entardecer", "à noite", "de madrugada"]
traffic_density = ["trânsito leve", "trânsito moderado", "trânsito intenso", "engarrafamento"]

locations = [
    "em uma rua movimentada", "no coração de uma grande avenida",
    "num bairro residencial tranquilo", "próximo de um parque verdejante",
    "perto de um shopping movimentado", "na periferia da cidade",
    "junto a uma escola", "perto de uma estação de metro"
]

object_behaviors = [
    "em movimento rápido", "parados no semáforo", "aguardando para atravessar",
    "estacionados", "circulando lentamente", "cruzando a via"
]

# Dicionário de números para enriquecer os objetos
numeros = {1: "um", 2: "dois", 3: "três", 4: "quatro", 5: "cinco"}

# Função para inserir, opcionalmente, um pequeno erro de digitação (baixa probabilidade)
def introduzir_erro(texto, prob=0.1):
    if random.random() < prob and len(texto) > 3:
        pos = random.randint(0, len(texto) - 2)
        return texto[:pos] + texto[pos+1] + texto[pos] + texto[pos+2:]
    return texto

# Templates revisados para uma redação mais natural (Português de Portugal)
templates = [
    "Num cenário {location}, num dia {weather} {time}, observa-se {objects} que se encontram {behavior}, com {traffic} ao fundo.",
    "Observa-se {objects} que se encontram {behavior} em {location}, num dia {weather} {time}, acompanhado de {traffic}.",
    "A cena revela {objects}, atualmente {behavior}, situados em {location} sob condições de clima {weather} {time}, marcados por {traffic}.",
    "{weather} {time} em {location}: nota-se {objects} a {behavior}, enquanto o ambiente exibe {traffic}.",
    "No contexto de {location}, destacam-se {objects} que se encontram {behavior} num período {weather} {time}, com evidências de {traffic}.",
    "Sob o céu {weather} {time}, {objects} podem ser vistos {behavior} em {location}, compondo uma paisagem com {traffic}.",
    "Durante um instante {weather} {time} em {location}, percebe-se a presença de {objects} que estão {behavior}, evidenciando {traffic}.",
    "Num ambiente {location}, num dia {weather} {time}, saltam aos olhos {objects} que se encontram {behavior}, com uma característica de {traffic}.",
    "Enquanto o {weather} {time} se desenrola em {location}, nota-se {objects} a {behavior}, realçados pelo cenário com {traffic}.",
    "Em {location}, sob um clima {weather} {time}, pode-se observar {objects} que se encontram {behavior}, formando um quadro de {traffic}.",
    "Durante uma tarde {weather} {time} em {location}, destaca-se a presença de {objects} que se encontram {behavior}, com {traffic} evidenciado ao fundo.",
    "Num cenário de {weather} {time} em {location}, {objects} aparecem {behavior}, enquanto o ambiente se caracteriza por {traffic}.",
    "Sob o olhar atento de quem passa por {location}, num dia {weather} {time}, {objects} mostram-se {behavior} e complementam o ambiente com {traffic}."
]

# Função para gerar objetos (rótulos) com variação
def generate_objects():
    # Determina o número de objetos usando uma distribuição de Poisson (lambda=3)
    num_objects = int(np.random.poisson(lam=3))
    num_objects = max(2, min(num_objects, len(combined_labels)))
    selected = random.sample(combined_labels, k=num_objects)
    
    enriched_objects = []
    for obj in selected:
        # Com probabilidade de 40%, prefixa com um numeral aleatório (entre 1 e 5)
        if random.random() < 0.4:
            n = random.randint(1, 5)
            objeto_mod = f"{numeros[n]} {obj}"
            objeto_mod = introduzir_erro(objeto_mod, prob=0.05)
            enriched_objects.append(objeto_mod)
        else:
            enriched_objects.append(obj)
    if len(enriched_objects) == 2:
        return " e ".join(enriched_objects)
    else:
        return ", ".join(enriched_objects[:-1]) + " e " + enriched_objects[-1]

# Função para gerar os dados sintéticos enriquecidos
def generate_synthetic_data(num_samples=1000):
    synthetic_data = []
    for _ in range(num_samples):
        objects = generate_objects()
        weather = random.choice(weather_conditions)
        time = random.choice(time_of_day)
        location = random.choice(locations)
        traffic = random.choice(traffic_density)
        behavior = random.choice(object_behaviors)
        
        template = random.choice(templates)
        description = template.format(
            objects=objects,
            weather=weather,
            time=time,
            location=location,
            traffic=traffic,
            behavior=behavior
        )
        
        description = introduzir_erro(description, prob=0.05)
        description = "startseq " + description.strip() + " endseq"
        
        # Para os rótulos, removemos eventuais prefixos numéricos para manter consistência
        labels = [
            " ".join(word for word in obj.split() if word not in numeros.values())
            for obj in objects.replace(",", " e ").split(" e ")
        ]
        synthetic_example = {
            "labels": [label.strip() for label in labels],
            "description": description,
            "origem": "sintético"
        }
        synthetic_data.append(synthetic_example)
    return synthetic_data

if __name__ == "__main__":
    # Gerar 10 exemplos para visualização
    exemplos = generate_synthetic_data(10)
    for i, exemplo in enumerate(exemplos, 1):
        print(f"Exemplo {i}:")
        print("Labels:", exemplo["labels"])
        print("Description:", exemplo["description"])
        print("-" * 80)
    
    # Salvar os exemplos num ficheiro JSON
    with open('./data/synthetic_and_real_data.json', 'w', encoding='utf-8') as f:
        json.dump(exemplos, f, indent=4, ensure_ascii=False)
    
    print("Exemplos gerados e salvos com sucesso!")
