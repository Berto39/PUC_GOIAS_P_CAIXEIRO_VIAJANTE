import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
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

# Plotar a rota em um mapa do Brasil
def plotar_rota(rota):
    # Configuração do mapa
    plt.figure(figsize=(12, 10))
    mapa = Basemap(
        projection='merc',
        llcrnrlat=-35.0,
        urcrnrlat=5.0,
        llcrnrlon=-75.0,
        urcrnrlon=-34.0,
        resolution='i'
    )
    mapa.drawcoastlines()
    mapa.drawcountries()
    mapa.drawstates()
    mapa.drawmapboundary(fill_color='aqua')
    mapa.fillcontinents(color='lightgreen', lake_color='aqua')

    # Adicionar as cidades ao mapa
    for cidade in cidades.keys():
        lat, lon = cidades[cidade]
        x, y = mapa(lon, lat)
        mapa.plot(x, y, 'bo', markersize=5)
        plt.text(x, y, cidade, fontsize=8)  # Diminuir o tamanho da fonte

    # Adicionar as rotas ao mapa
    for i in range(len(rota) - 1):
        cidade_atual = list(cidades.keys())[rota[i]]
        proxima_cidade = list(cidades.keys())[rota[i + 1]]
        lat1, lon1 = cidades[cidade_atual]
        lat2, lon2 = cidades[proxima_cidade]
        x1, y1 = mapa(lon1, lat1)
        x2, y2 = mapa(lon2, lat2)
        mapa.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

    # Fechar o ciclo, conectando a última cidade à primeira
    cidade_atual = list(cidades.keys())[rota[-1]]
    proxima_cidade = list(cidades.keys())[rota[0]]
    lat1, lon1 = cidades[cidade_atual]
    lat2, lon2 = cidades[proxima_cidade]
    x1, y1 = mapa(lon1, lat1)
    x2, y2 = mapa(lon2, lat2)
    mapa.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

    # Adicionar título e legendas aos eixos
    plt.title("Melhor Rota Encontrada")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

# Exemplo de plotagem
plotar_rota(melhor_individuo)