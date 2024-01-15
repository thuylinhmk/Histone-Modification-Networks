# %%
import pandas as pd
import numpy as np
import os
from faiss_cluster import *
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
import seaborn as sns
from graph_vis import *


# %%
chen_var = pd.read_csv('../chen_variants.hg38.vcf',sep="\t", comment='#')
chen_var = chen_var.sort_values(['CHR', 'hg38_POS'], ignore_index=True)

# %%
chen_var.query('ASSAY != "gene"')

# %%
chen_var.TISSUE.unique(), chen_var.ASSAY.unique()

# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=2, subplot_titles=("Pvalue", 'Slop'))

fig.add_trace(
    go.Box(y=chen_var['PVALUE'], name='pvalue'),
    row=1, col=1, 
)

fig.add_trace(
    go.Box(y=chen_var['SLOPE'], name='slope'),
    row=1, col=2, 
)

fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
fig.show()

# %%
gencode_v29 = pr.read_gtf('../region/gencode.v29.annotation.gtf')

# %%
gencode_v29.Feature.unique()

# %% [markdown]
# --------------------------------
# GTEX: 
# - cis eQLT: variants within +-1Mb of the TSS.
# - trans eQLT: variants within +-5Mb of the TSS

# %%
#### GTEX variant files """
files = os.listdir("../gtex/")
first_f = True
for f in files:
    if '.egenes' in f:
        tissue = f.split('.v8')[0]
        #print(tissue)
        if first_f:
            gene_tested = pd.read_csv(f'../gtex/{f}', sep='\t', compression='gzip', engine='c')
            gene_tested['Tissue'] = [tissue]*len(gene_tested)
            first_f=False
        else:
            reading = pd.read_csv(f'../gtex/{f}', sep='\t', compression='gzip', engine='c')
            reading['Tissue'] = [tissue]*len(reading)
            gene_tested = pd.concat([gene_tested, reading], axis=0)
#gene_tested = gene_tested.query("qval <= 0.05")

# %% [markdown]
# gtex: 
# - num_var = number of variants in cis-window
# - slope = regression slop (normalized effect size - nes in *.signif files)

# %%
gene_tested.query(f'variant_pos in {chen_var.hg38_POS.tolist()} and Tissue.str.contains("Blood")')

# %%
overlaps_gtex_chen = gene_tested.merge(chen_var, left_on=['chr', 'variant_pos'], right_on=['CHR', 'hg38_POS'])

# %%
overlaps_gtex_chen.groupby('TISSUE').size()

# %%
chen_var.groupby('TISSUE').size()

# %%
blood_cell = pd.read_csv('../Histone_data_tsv/blood_cell.csv')

# %% [markdown]
# ### network

# %%
#### anndata for blood cell
target = ['H3K27ac', 'H3K4me1']
count_files=os.listdir('../counting_csv_out/gencode_v29_tss/')
count_files= [x for x in count_files if ('chrX' not in x and 'chrY' not in x)]

# %%
target_dict = {}
for h in target:
    target_dict[h] = {}
    for f in count_files:
        if h in f:
            c = f.split('_')[1]
            peak_count = read_feather_hm(f'../counting_csv_out/gencode_v29_tss/{f}', fix_id=True)
            #peak_count.loc[:, 'Target'] = t
            target_dict[h][c] = peak_count
            print(h, c, target_dict[h][c].shape)

# %%
duplicate = chen_var[chen_var.duplicated(['CHR', 'hg38_POS', 'TISSUE'])]

# %%
duplicate

# %%
annotation = gencode_v29.df.query('Feature == "gene"').sort_values('level').drop_duplicates('gene_name', keep='first')

# %% [markdown]
# ######## assume the change in histone correlated in gene expression ########

# %%
blood_cell.groupby('Tissue').size()

# %%
chrom1 = []
for h in target:
    chrom1.append(target_dict[h]['chr1'])

chrom1_count = pd.concat(chrom1, axis=0)
cols = chrom1_count.sum(0)
masked = cols[cols>0]

chrom1_count = chrom1_count.loc[:, masked.keys()]



# %%
for h in target:
    for tissue in blood_cell.Tissue.unique():
        if 'neutr' in tissue:
            continue
        else:
            print(h, tissue)
            tissue_assay = blood_cell.query(f'Tissue == "{tissue}" and `Target of assay` == "{h}"')
            tissue_count = chrom1_count.loc[tissue_assay.Code, :].astype(bool)
            checking = (tissue_count == 1).mean(0)
            change_cols = np.where(checking <0.3)
            print(change_cols)
            chrom1_count.loc[tissue_assay.Code, np.take(chrom1_count.columns.tolist(), change_cols[0])] = 0

# %%
var = list_domain_todf(chrom1_count.columns.tolist())
obs = blood_cell.set_index('Code')

chrom1_blood_anndata = anndata.AnnData(X=chrom1_count.to_numpy(), var=var, obs=obs)

# %%
#### avg
avg_sv = []
for hm in target:
    hm_assay = blood_cell.query(f'`Target of assay` == "{hm}"').Code.tolist()
    peak_hm = chrom1_count.loc[hm_assay,:]
    peak_hm_filter = peak_hm.astype('bool').astype('int')
    data_path = rf'../data/blood_cell/{hm}/'
    signal = average_signalValue_pyranges(peak_hm, data_path, average=True, prefix=None)
    signal = signal*peak_hm_filter
    avg_sv.append(signal)

# %%
chrom1_blood_anndata

# %%
avg_sv = pd.concat(avg_sv, axis=0)
print(avg_sv.shape, chrom1_count.shape)
avg_sv = avg_sv.loc[chrom1_count.index, chrom1_count.columns]

# %%
chrom1_blood_anndata.layers['average_signal'] = avg_sv

# %%
#chrom1_blood_anndata.write_h5ad('blood_anndata_chr1.h5ad')

# %%
def filter_nonzero(data):
    col = data.sum(0)
    masked = col[col>0]
    df = data.loc[:,masked.keys()]
    return df

# %%
chrom='chr1'
target=['H3K4me1']
for hm in target:
    tissue = chrom1_blood_anndata.obs.query(f'`Target of assay` == "{hm}"')['Tissue'].unique()
    for t in tissue:
        if 'neutr' in t: 
            continue
        subset = chrom1_blood_anndata[(chrom1_blood_anndata.obs['Target of assay'] == hm ) &
                            (chrom1_blood_anndata.obs['Tissue'] == t)]
        signal = subset.to_df('average_signal')
        filtered_signal = filter_nonzero(signal)
        print(f'{hm}-{t} after filter regions with no signal: {filtered_signal.shape}')
        rank = rank_transformation(filtered_signal)
        corr = correlation_cal(rank, 'pearsonr')
        adjacency_matrix = get_adjacency_matrix(corr, 0.7, keep_diag=False).round(8)
        #create network
        network = HMNetwork(name=f'{hm}_{t}')
        network.from_adjacency(adjacency_matrix)
        #network feature
        network.leidenPartition(weight_name='weight', name='leiden_CPM', partition_func=la.CPMVertexPartition, resolution_parameter=0.01)  
        
        network._eigenvector_centrality(weight_name='weight', col_name='evcent_all')
        
        network._eigenvector_centrality(weight_name='weight', col_name='evcent_CPM', for_partition=True, partition_col='leiden_CPM')

        network._vertex_strength('weight')
        
        network.leidenPartition(weight_name='weight', name='leiden_modularity', partition_func=la.ModularityVertexPartition, use_abs_weight=True) 
        
        network._eigenvector_centrality(weight_name='weight', col_name='evcent_modularity', for_partition=True, partition_col='leiden_modularity')
        
        vertex_pearson = network.get_node_dataframe()
        vertex_pearson['degree'] = network._igraph.degree(loops=False)
        vertex_pearson['average_strength'] = vertex_pearson['strength']/vertex_pearson['degree']

        adjacency_matrix.reset_index().to_feather(f'network_dataframe_majortissue/adjacency_matrix/blood_{hm}_{t}_{chrom}_adjacency_matrix.feather')
        vertex_pearson.reset_index().to_feather(f'network_dataframe_majortissue/vertex_df/blood_{hm}_{t}_{chrom}_vertex.feather')
        print(f'Done {hm}-{t}')


