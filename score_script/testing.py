import networkx as nx
import igraph as ig
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import leidenalg as la

data1 = pd.read_feather('../processing_data/H3K27ac/cluster/h3k27ac_full_5cluster_top1000.feather').set_index('index')

data1_region_info = data1[['Cluster', 'Type']]
data1.drop(['Cluster', 'Type'], inplace=True, axis=1)

scaler = MinMaxScaler()
scaler.fit(data1.T)
data_scaled = pd.DataFrame(scaler.transform(data1.T).T, index=data1.index, columns=data1.columns)

print('corr cal')
scaled_corr, _ = spearmanr(data_scaled.T)
scaled_corr = pd.DataFrame(scaled_corr, index=data_scaled.index, columns=data_scaled.index)

def get_adjacency_matrix(data, threshold, absolute):
    df = data.copy()
    if absolute:
        df[abs(df) < threshold] = 0
    else:
        df[df < threshold] = 0 
    np_df = df.values
    np.fill_diagonal(np_df, 0)
    df = pd.DataFrame(np_df, index=df.index, columns=df.columns)    
    
    return df
        
print('network build')        
scaled_adjacency = get_adjacency_matrix(scaled_corr, 0.5, True)
nx_scaled = nx.from_pandas_adjacency(scaled_adjacency)
scaled_igraph = ig.Graph.from_networkx(nx_scaled)
scaled_igraph.vs['cluster'] = data1_region_info.loc[scaled_igraph.vs['_nx_name'],:].Cluster.tolist()
scaled_igraph.vs['type'] = data1_region_info.loc[scaled_igraph.vs['_nx_name'],:].Cluster.tolist()
scaled_igraph.vs['name'] = scaled_igraph.vs['_nx_name']

print('leiden partition')
scaled_partition = la.find_partition(scaled_igraph, 
                                   la.CPMVertexPartition,
                                   weights='weight',
                                   n_iterations=-1,
                                   seed=218,
                                   resolution_parameter=0.01)
scaled_igraph.vs['leiden_comm'] = scaled_partition.membership

scaled_igraph.es['abs_weight'] = list(map(abs,scaled_igraph.es['weight']))

print('evcent')
scaled_igraph.vs['evcent_all'] = scaled_igraph.eigenvector_centrality(weights='abs_weight', directed=False)

print('betweenness')
scaled_igraph.vs['betweenness_all'] = scaled_igraph.betweenness(directed=False,
                                                               weights='abs_weight')

scaled_subgraph = scaled_partition.subgraphs()
for graph in scaled_subgraph:
    e_score = graph.eigenvector_centrality(directed=False,
                                                 weights='abs_weight')
    b_score = graph.betweenness(directed=False,
                                weights='abs_weight')
    for index, node in enumerate(graph.vs):
            scaled_igraph.vs.find(name=node['_nx_name'])['evcent_comm'] = e_score[index]
            scaled_igraph.vs.find(name=node['_nx_name'])['betweenness_comm'] = e_score[index]

print('write file')
vertex_df = scaled_igraph.get_vertex_dataframe()
vertex_df.reset_index().to_feather('score_full_h3k27ac_subset5k.feather')

print('all done')


