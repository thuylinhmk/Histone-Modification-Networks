
import pandas as pd
import numpy as np
import os
import pyranges as pr
from PyWGCNA import WGCNA
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import igraph as ig
import leidenalg as la
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import scanpy
import colorcet as cc
import seaborn as sns
import powerlaw
import scipy.stats
from gprofiler import GProfiler
from typing import Union
import re
from collections import Counter


# %%
def list_domain_todf(list_domain):
    chrom = [x.split(':')[0] for x in list_domain]
    start = [re.split(':|-', x)[1] for x in list_domain]
    end = [re.split(':|-', x)[2] for x in list_domain]
    return pd.DataFrame({'Chromosome': chrom, 
                         'Start': start, 'End':end}, index=list_domain)

# %%
def filter_allzero(df):
    cols = df.columns
    df_sum = df.sum(0)
    masked = cols[df_sum>0]
    return df[masked]

# %%
def correlation_cal(data, test='spearmanr', fillna=0, pval=False):
    if test == 'spearmanr':
        if not pval:
            scaled_corr, _ = spearmanr(data)
        else:
            scaled_corr, pval = spearmanr(data)
    elif test == 'pearsonr':
        if not pval:
            scaled_corr = np.corrcoef(data.to_numpy().T)
        else:
            raise ValueError("not support pval for pearsonr")
    else:
        raise ValueError('Not supported: choose between pearsonr or spearmanr')
         

    np.fill_diagonal(scaled_corr, 1)
    scaled_corr = pd.DataFrame(scaled_corr, index=data.columns, columns=data.columns)
    scaled_corr.fillna(fillna, inplace=True)
    if not pval:
        return scaled_corr
    else:
        return (scaled_corr, pval)

# %%
def rank_transformation(data, scaled='MinMax'):
    """ Rank transformation and scale across assays
        Arguments:
            data (pd.DataFrame): samples x features
            scaled (str): type of scaler to scale data across sample
                (default: MinMax)
                None: for not using scale     
    """
    rank_transform = data.to_numpy().argsort().argsort()
    if not scaled:
        return pd.DataFrame(rank_transform, 
                                    index=data.index,
                                    columns=data.columns)
    else:
        scaler = MinMaxScaler()
        scaler.fit(rank_transform)
        df = pd.DataFrame(scaler.transform(rank_transform), 
                                        index=data.index,
                                        columns=data.columns) 
        return df

# %%
def powerlaw_fit_plot(corr_df, 
                    power, 
                    title: str,
                    threshold=None, 
                    sfti: Union['connectivity', 'degree'] = 'connectivity'):
    unsigned_corr = abs(corr_df.to_numpy())
    TOMsim_signed = unsigned_corr**power

    if threshold: 
        q=np.quantile(upper, 0.99)
        print(q3)
        fil = np.where(TOMsim_signed<q, 0, TOMsim_signed)
    else:
        fil=TOMsim_signed

    fil=pd.DataFrame(TOMsim_signed, index=corr_df.index, columns=corr_df.columns)
    graph = ig.Graph.Weighted_Adjacency(fil, 'upper', loops=False)
    print(f'{graph.vcount()} nodes, {graph.ecount()} edges')
    if sfti == 'connectivity':
        print(f'Mean connectivity: {np.sum(graph.strength(weights="weight"))/graph.vcount()}')
        _sequence = sorted(graph.strength(weights='weight'), reverse=True)
        _sequence = np.round(_sequence)
    elif sfti == 'degree':
        print(f'Mean connectivity: {np.sum(graph.degree())/graph.vcount()}')
        _sequence = sorted(graph.degree(), reverse=True)
    
    fig, ax = plt.subplots(figsize=(10,7))
    plt.yscale('log')
    plt.xscale('log')

    fit = powerlaw.Fit(_sequence)
    print(fit.power_law.alpha, fit.power_law.sigma, fit.xmin)
    print(fit.distribution_compare('power_law',  'lognormal', normalized_ratio=True))
    deg = _sequence
    deg_distri = Counter(deg)
    x=[]
    y=[]
    for i in sorted(deg_distri):
            x.append(i)
            y.append(deg_distri[i]/graph.vcount())
    ax.plot(x, y,'ro')
    fig=fit.plot_pdf(color='r', label='data', ax=ax)
    fit.power_law.plot_pdf(color='b', linestyle='-', linewidth=1, label='powerlaw fit', ax=ax)
    fit.lognormal.plot_pdf(color='g', linestyle='-', linewidth=1, label='lognormal fit', ax=ax)
    fig.legend(fontsize=13)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=22)
    plt.xlabel(f'{sfti} $k$', fontsize=22)
    plt.ylabel('$P(k)$', fontsize=22)
    plt.title(title)
    ax.text(0.7, 1.0, f'$\\alpha$={(round(fit.alpha, 2))}, xmin={fit.xmin}', transform=ax.transAxes, fontdict={'size': 13})
    plt.savefig(f'{title}.png')


# %%
def module_trait_relationships_heatmap(pywgcna_obj,
                                        file_name: str,
                                        plot_title: str,
                                        metaData=['Tissue'],
                                        show=True,
                                           ):
        """
        plot topic-trait relationship heatmap

        :param metaData: traits you would like to see the relationship with topics (must be column name of datExpr.obs)
        :type metaData: list
        :param show: indicate if you want to show the plot or not (default: True)
        :type show: bool
        :param file_name: name and path of the plot use for save (default: topic-traitRelationships)
        :type file_name: str
        """
        datTraits = pywgcna_obj.getDatTraits(metaData)
        pvalue = pywgcna_obj.moduleTraitPvalue.loc[:, datTraits.columns]
        cor = pywgcna_obj.moduleTraitCor.loc[:, datTraits.columns]

        fig, ax = plt.subplots(figsize=(max(20, int(pvalue.shape[0] * 1.5)),
                                        pvalue.shape[1] * 1.5), facecolor='white')
        # names
        xlabels = []
        for label in pywgcna_obj.MEs.columns:
            xlabels.append(label[2:].capitalize() + '(' + str(sum(pywgcna_obj.datExpr.var['moduleColors'] == label[2:])) + ')')
        ylabels = datTraits.columns

        # Loop over data dimensions and create text annotations.
        tmp_cor = cor.T.round(decimals=2)
        tmp_pvalue = pvalue.T.round(decimals=3)
        labels = (np.asarray(["{0}\n({1})".format(cor, pvalue)
                              for cor, pvalue in zip(tmp_cor.values.flatten(),
                                                     tmp_pvalue.values.flatten())])) \
            .reshape(cor.T.shape)

        sns.set(font_scale=1.5)
        res = sns.heatmap(cor.T, annot=labels, fmt="", cmap='RdBu_r',
                          vmin=-1, vmax=1, ax=ax, annot_kws={'size': 20, "weight": "bold"},
                          xticklabels=xlabels, yticklabels=ylabels)
        res.set_xticklabels(res.get_xmajorticklabels(), fontsize=20, fontweight="bold", rotation=90)
        res.set_yticklabels(res.get_ymajorticklabels(), fontsize=20, fontweight="bold")
        plt.yticks(rotation=0)
        ax.set_title(plot_title,
                     fontsize=30, fontweight="bold")
        ax.set_facecolor('white')
        fig.tight_layout()

        plt.savefig(f'{file_name}.png')

# %% [markdown]
# --------------

# %%
mainpath='network_dataframe_majortissue' #path pls 

# %%
target = ['H3K4me3', 'H3K27me3', 'H3K4me1', 'H3K27ac', 'H3K9me3']

# %%
adata_files = [f for f in os.listdir(mainpath) if '.h5ad' in f]

# %%
adata_files

# %%
color_p = cc.glasbey_warm

# %%
for f in adata_files:
    adata = scanpy.read_h5ad(f'{mainpath}/{f}')
    chrom = re.split('_|\.', f)[3]
    if chrom not in ['chr1','chr2', 'chr4', 'chr5', 'chr18', 'chr17', 'chr13', ]:
	    continue
    for h in target:
        print(h, chrom)
        hm = adata[adata.obs['Target of assay']==h]
        hm_info = hm.obs
        hm = hm.to_df('average_signal')
        hm = filter_allzero(hm)

        hm_info = adata.obs.query(f'`Target of assay` == "{h}"')

        rank_full = rank_transformation(hm)
        corr_full = correlation_cal(rank_full, 'pearsonr')
        
        power, _ = WGCNA.pickSoftThreshold(rank_full, 'unsigned')

        powerlaw_fit_plot(corr_df=corr_full, power=power, title=f'Power fit {h} - {chrom}')

        # pyWGCNA_hm = WGCNA(name=h, 
        #                       species='homo sapiens', 
        #                       geneExp=rank_full,
        #                       networkType='unsigned',)
        # pyWGCNA_hm.findModules()

        # pyWGCNA_hm.updateSampleInfo(hm_info)

        # tissue_color = dict(zip(hm_info.Tissue.unique().tolist(), color_p))

        # pyWGCNA_hm.setMetadataColor('Tissue', tissue_color)
        # pyWGCNA_hm.analyseWGCNA()

        # module_trait_relationships_heatmap(pywgcna_obj=pyWGCNA_hm, plot_title=f"Module-Trait relationships heatmap for {h} - {chrom}", file_name=f'module_trait_{h}_{chrom}')

        # list_regions = []
        # for name in pyWGCNA_hm.getModuleName():
        #     list_regions.append(pyWGCNA_hm.top_n_hub_genes(name, n=-1))
        # list_regions_df = pd.concat(list_regions, axis=0)
        # list_regions_df.reset_index().to_feather(f'{h}_{chrom}_wgcna_region.feather')


