
import pandas as pd
import numpy as np
import os
import pyranges as pr
from PyWGCNA import WGCNA
import os
from gprofiler import GProfiler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import igraph as ig
import leidenalg as la
import plotly.express as px
import re
import plotly.graph_objects as go
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import scanpy
import seaborn as sns
import colorcet as cc
from simple_function import *




# %%
def filter_allzero(df):
    cols = df.columns
    df_sum = df.sum(0)
    masked = cols[df_sum>0]
    return df[masked]

# %%


def processing(file):
    q=[]
    adata = scanpy.read_h5ad(f'{mainpath}/{f}')
    chrom = re.split('_|\.', f)[3]
    for h in target:
        print(h)
        hm = adata[adata.obs['Target of assay']==h]
        hm_info = hm.obs
        hm = hm.to_df('average_signal')
        hm = filter_allzero(hm)

        hm_info = adata.obs.query(f'`Target of assay` == "{h}"')

        rank_full = rank_transformation(hm)
        corr_full = correlation_cal(rank_full, 'pearsonr')
        print(corr_full.shape)
        
        
        power, _ = WGCNA.pickSoftThreshold(rank_full, 'unsigned')

        unsigned=abs(corr_full)**power
        network = ig.Graph.Weighted_Adjacency(unsigned, 'upper', loops=False)
        partition = la.find_partition(network, la.ModularityVertexPartition, weights='weight', n_iterations=10, seed=218)
        q.append((h, chrom, np.max(partition.membership)+1, partition.quality()))
    module_leiden = pd.DataFrame(q, columns = ['Histone', 'Chromosome', 'Number of module', 'Quality score - Modularity'])
    module_leiden.to_feather(f'leiden_modularity_{chrom}.feather')



if __name__ == 'main':
        # %%
    target = ['H3K27me3', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K9me3']

    # %%
    mainpath = 'network_dataframe_majortissue'

    # %%
    adata_files = os.listdir('network_dataframe_majortissue')