import pandas as pd
import numpy as np
import os
import pyranges as pr
import PyWGCNA
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
import powerlaw
from simple_function import *
from PyWGCNA import WGCNA
from collections import Counter

sf_fig_folder = 'project/Figure/scale_free'
#h5ad file
adatas_file = [x for x in os.listdir('network_dataframe_majortissue/') if '.h5ad' in x]

def filter_allzero(df):
    cols = df.columns
    df_sum = df.sum(0)
    masked = cols[df_sum>0]
    return df[masked]

for f in adatas_file:
    chrom = f.split('_')[3].split('.')[0]
    adata = scanpy.read_h5ad(f'network_dataframe_majortissue/{f}')
    target = adata.obs['Target of assay'].unique()
    for hm in target:
        tissue = adata.obs.query('`Target of assay` == @hm').Tissue.unique()
        for t in tissue:
            print(hm , t, chrom)
            sub = adata[(adata.obs['Target of assay'] == hm) & (adata.obs['Tissue']==t)].to_df('average_signal')
            test =filter_allzero(sub)
            rank = rank_transformation(test)
            corr = correlation_cal(rank, 'pearsonr')

            unsigned_corr = abs(corr.to_numpy())
            TOMsim_signed = unsigned_corr**9
            upper = TOMsim_signed[np.triu_indices(TOMsim_signed.shape[0], k = 1)]
            q3=np.quantile(upper, 0.99)
            print(q3)
            fil = np.where(TOMsim_signed<q3, 0, TOMsim_signed)
            fil=pd.DataFrame(fil, index=corr.index, columns=corr.index)
            graph = ig.Graph.Weighted_Adjacency(fil, 'upper', loops=False)
            print(graph.vcount(), graph.ecount())
            print(f'Mean connectivity: {np.sum(graph.strength(weights="weight"))/graph.vcount()}')
            degree_sequence = sorted(graph.degree(), reverse=True)

            fig, ax = plt.subplots(figsize=(10,7))
            plt.yscale('log')
            plt.xscale('log')

            fit = powerlaw.Fit(degree_sequence, discrete=True)
            print(fit.power_law.alpha, fit.power_law.sigma, fit.xmin)
            print(fit.distribution_compare('power_law',  'lognormal', normalized_ratio=True))
            deg = degree_sequence
            deg_distri = Counter(deg)
            x =[]
            y=[]
            for i in sorted(deg_distri):
                if i>fit.xmin:
                    x.append(i)
                    y.append(deg_distri[i]/graph.vcount())
            ax.plot(x,y,'ro')
            fig=fit.plot_pdf(color='r', label='data', ax=ax)
            fit.power_law.plot_pdf(color='b', linestyle='-', linewidth=1, label='powerlaw fit', ax=ax)
            fit.lognormal.plot_pdf(color='g', linestyle='-', linewidth=1, label='lognormal fit', ax=ax)
            fig.legend(fontsize=13)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=22)
            plt.xlabel('degree $k$', fontsize=22)
            plt.ylabel('$P(k)$', fontsize=22)
            plt.title(f'Log-Log degree distribution: ({hm} - {t} - {chrom})')
            ax.text(0.75, 0.7, f'$\\alpha$={(round(fit.alpha, 2))}, xmin={fit.xmin}', transform=ax.transAxes, fontdict={'size': 13})
            plt.savefig(f'{sf_fig_folder}/{hm}_{t}{chrom}_powerlawfit_unsigned.png')
            