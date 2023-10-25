import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler #para normalizar los valores

path = 'data/'
dominio_encoder = {0:'Semi urbano', 1:'Rural', 2:'Urbano'}

# CREAR REDES
### Nacional
df = pd.read_csv(path+'pesos_nac.csv', na_values=[np.nan, 0], index_col='Unnamed: 0')
df = df.abs()
for col in df.max().loc[df.max()>1].index: #normalizo las columnas con valores mayores al rango (-1,1)
    df.loc[col, col] = -(df[col].max())
    df[col] = MinMaxScaler().fit_transform(np.array(df[col]).reshape(-1, 1))
    df.loc[col, col] = np.nan
G_2 = nx.empty_graph(0, nx.MultiDiGraph()) #initialize an empty weighted directed graph
for row in df.index:
    for col in df.columns:
        if not np.isnan(df.loc[row, col]) and (df.loc[row, col] > 0):
            G_2.add_edge(row, col, weight=df.loc[row, col])    
### Por dominio
dominio = [nx.empty_graph(0, nx.MultiDiGraph()), nx.empty_graph(0, nx.MultiDiGraph()), nx.empty_graph(0, nx.MultiDiGraph())]
for i in range(3):
    #cargo dataset
    pesos = pd.read_csv(path+f'pesos_dom_{i}.csv', na_values=[np.nan, 0], index_col='Unnamed: 0')
    pesos = pesos.abs()
    for col in pesos.max().loc[pesos.max()>1].index: #normalizo las columnas con valores mayores al rango (-1,1)
        pesos.loc[col, col] = -(pesos[col].max())
        pesos[col] = MinMaxScaler().fit_transform(np.array(pesos[col]).reshape(-1, 1))
        pesos.loc[col, col] = np.nan
    #creo las aristas dirigidas pesadas del grafo
    for row in pesos.index:
        for col in pesos.columns:
            if not np.isnan(pesos.loc[row, col]) and (df.loc[row, col] > 0):
                dominio[i].add_edge(row, col, weight=pesos.loc[row, col])


# ELIMINAR LAS ARISTAS CON LOS VALORES MENORES A LOS PUNTOS DE CORTE
cut_point = [0.045, 0.048, 0.06, 0.1]
### Nacional
le_ids = list(e[:2] for e in filter(lambda e: e[2] <= cut_point[-1], (e for e in G_2.edges.data('weight'))))
G_2.remove_edges_from(le_ids)
### Por dominio
for i in range(3):
    le_ids = list(e[:2] for e in filter(lambda e: e[2] <= cut_point[i], (e for e in dominio[i].edges.data('weight'))))
    dominio[i].remove_edges_from(le_ids)


# OBTENER LAS MÉTRICAS
## Pagerank
pagerank_values = pd.DataFrame(index=df.columns)
pagerank_ranking = pd.DataFrame(index=df.columns)
pagerank_sorted = pd.DataFrame()
pagerank_values['nacional'] = [nx.pagerank_numpy(G_2)[pr] for pr in df.columns] 
pagerank_ranking['nacional'] = np.nan
for i,tupla in enumerate(sorted(nx.pagerank_numpy(G_2).items(), key=lambda x: x[1], reverse=True)):
    pagerank_ranking.loc[tupla[0],'nacional'] = i
pagerank_sorted['nacional'] = [k for k,v in sorted(nx.pagerank_numpy(G_2).items(), key=lambda x: x[1], reverse=True)]
for g in range(3):
    pagerank_values[f'dom_{g}'] = np.insert([nx.pagerank_numpy(dominio[g])[pr] for pr in pesos.columns], 1, np.nan)
    pagerank_ranking[f'dom_{g}'] = np.nan
    for i,tupla in enumerate(sorted(nx.pagerank_numpy(dominio[g]).items(), key=lambda x: x[1], reverse=True)):
        pagerank_ranking.loc[tupla[0],f'dom_{g}'] = i+1
    pagerank_sorted[f'dom_{g}'] = np.insert([k for k,v in sorted(nx.pagerank_numpy(dominio[g]).items(), key=lambda x: x[1], reverse=True)], 42, np.nan)
pagerank_values.to_csv(path+'pagerank_values.csv')
pagerank_ranking.to_csv(path+'pagerank_ranking.csv')
pagerank_sorted.to_csv(path+'pagerank_sorted.csv', index=False)

# Centralidad de grado
degree_centrality_values = pd.DataFrame(index=df.columns)
degree_centrality_ranking = pd.DataFrame(index=df.columns)
degree_centrality_sorted = pd.DataFrame()
degree_centrality_values['nacional'] = [nx.degree_centrality(G_2)[pr] for pr in df.columns]
degree_centrality_ranking['nacional'] = np.nan
for i,tupla in enumerate(sorted(nx.degree_centrality(G_2).items(), key=lambda x: x[1], reverse=True)):
    degree_centrality_ranking.loc[tupla[0],'nacional'] = i
degree_centrality_sorted['nacional'] = [k for k,v in sorted(nx.degree_centrality(G_2).items(), key=lambda x: x[1], reverse=True)]
for g in range(3):
    degree_centrality_values[f'dom_{g}'] = np.insert([nx.degree_centrality(dominio[g])[dc] for dc in pesos.columns], 1, np.nan)
    degree_centrality_ranking[f'dom_{g}'] = np.nan
    for i,tupla in enumerate(sorted(nx.degree_centrality(dominio[g]).items(), key=lambda x: x[1], reverse=True)):
        degree_centrality_ranking.loc[tupla[0],f'dom_{g}'] = i+1
    degree_centrality_sorted[f'dom_{g}'] = np.insert([k for k,v in sorted(nx.degree_centrality(dominio[g]).items(), key=lambda x: x[1], reverse=True)], 42, np.nan)
degree_centrality_values.to_csv(path+'degree_centrality_values.csv')
degree_centrality_ranking.to_csv(path+'degree_centrality_ranking.csv')
degree_centrality_sorted.to_csv(path+'degree_centrality_sorted.csv', index=False)
