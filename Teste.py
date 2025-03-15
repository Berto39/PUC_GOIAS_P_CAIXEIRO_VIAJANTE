import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import networkx as nx

# Leitura do arquivo CSV
df = pd.read_csv('Cidades.csv', sep=',')
print(df.columns)  # Verifica os nomes das colunas
cidades = {row['Cidades']: (row['Latitude'], row['Longitude']) for _, row in df.iterrows()}

num_cidades = len(cidades)

# Função para calcular a distância entre duas cidades
def calcular_distancia(cidade1, cidade2):
    nome_cidade1 = list(cidades.keys())[cidade1]
    nome_cidade2 = list(cidades.keys())[cidade2]
    coord1 = np.array(cidades[nome_cidade1])
    coord2 = np.array(cidades[nome_cidade2])
    return np.linalg.norm(coord1 - coord2)

# Função de aptidão: distância total da rota
def calcular_aptidao(rota):
    distancia_total = 0
    for i in range(len(rota) - 1):
        distancia_total += calcular_distancia(rota[i], rota[i + 1])
    distancia_total += calcular_distancia(rota[-1], rota[0])  # Volta à cidade inicial
    return (distancia_total,)  # Retorna uma tupla

# Configuração do DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(num_cidades), num_cidades)
toolbox.register("individuo", tools.initIterate, creator.Individuo, toolbox.indices)
toolbox.register("populacao", tools.initRepeat, list, toolbox.individuo)

toolbox.register("evaluate", calcular_aptidao)
toolbox.register("mate", tools.cxOrdered)  # Registrar a função de cruzamento
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Parâmetros do algoritmo genético
tamanho_populacao = 100
num_geracoes = 200
probabilidade_crossover = 0.7
probabilidade_mutacao = 0.2

# Inicialização da população
populacao = toolbox.populacao(n=tamanho_populacao)

# Execução do algoritmo genético
algorithms.eaSimple(populacao, toolbox, cxpb=probabilidade_crossover, mutpb=probabilidade_mutacao,
                    ngen=num_geracoes, verbose=True)

# Melhor indivíduo encontrado
melhor_individuo = tools.selBest(populacao, k=1)[0]
print("Melhor rota encontrada:", melhor_individuo)
print("Distância total:", calcular_aptidao(melhor_individuo)[0])

# Plotar a rota
def plotar_rota(rota):
    G = nx.Graph()

    # Dicionário de posições para NetworkX
    posicoes = {cidade: cidades[cidade] for cidade in cidades.keys()}

    # Adicionar as arestas ao grafo
    for i in range(len(rota) - 1):
        cidade_atual = list(cidades.keys())[rota[i]]
        proxima_cidade = list(cidades.keys())[rota[i + 1]]
        distancia = calcular_distancia(rota[i], rota[i + 1])
        G.add_edge(cidade_atual, proxima_cidade, weight=round(distancia, 2))

    # Fechar o ciclo, conectando a última cidade à primeira
    cidade_atual = list(cidades.keys())[rota[-1]]
    proxima_cidade = list(cidades.keys())[rota[0]]
    distancia = calcular_distancia(rota[-1], rota[0])
    G.add_edge(cidade_atual, proxima_cidade, weight=round(distancia, 2))

    # Configuração do gráfico
    plt.figure(figsize=(12, 10))
    nx.draw(
        G, 
        pos=posicoes, 
        with_labels=True, 
        node_size=2000, 
        node_color="lightblue", 
        font_size=10, 
        font_weight="bold"
    )
    # Adicionar rótulos às arestas (distâncias)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=posicoes, edge_labels=labels)

    # Adicionar título
    plt.title("Melhor Rota Encontrada")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis('on')
    plt.show()
    
# Exemplo de plotagem
plotar_rota(melhor_individuo)
