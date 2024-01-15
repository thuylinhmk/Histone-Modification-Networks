from pickle import FALSE
import numpy as np
import pandas as pd
import os
from simple_function import *
from graph_vis import *
import anndata
import plotly.express as px
import scipy 
import gzip
from concurrent.futures import ProcessPoolExecutor
import plotly.graph_objects as go
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FILTER = False
assay_info = []
target = ['H3K27me3', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K9me3']
#target = ['H3K27ac', 'H3K4me1']
for t in target:
    #file_info = pd.read_csv(f'../Histone_data_tsv/{t}_file_info.csv')
    file_info = pd.read_csv(f'../Histone_data_tsv/{t}_file_info.csv')
    file_info.columns = ['Code', 'Accession', 'Target of assay', 'Tissue', 'Biosample summary']
    assay_info.append(file_info)

assays = pd.concat(assay_info, axis=0)
#assays = pd.read_csv(f'../Histone_data_tsv/blood_cell.csv')
#assays = assays.query('~Tissue.str.contains("neutr")')
# %%
assays = assays.sort_values(['Target of assay', 'Tissue'], ignore_index=True)
if FILTER:
    # %%
    filters_assay = assays.groupby(['Target of assay', 'Tissue']).size()
    filters_assay = filters_assay[filters_assay>=10].reset_index().iloc[:, :2]
    filters_assay_tuples = list(filters_assay.itertuples(index=False, name=None))

    # %%
    filtered_df = assays[assays[['Target of assay', 'Tissue']].apply(tuple, axis=1).isin(filters_assay_tuples)]
else: 
    filtered_df = assays.copy()



def processing(chrom):
    count_files=os.listdir('../counting_csv_out/gencode_v29_tss/')
    count_files= [x for x in count_files if f'{chrom}_' in x and 'blood_cell' not in x]

    # %%
    target_dict = {}
    for f in count_files:
        t = f.split('_')[0]
        target_dict[t] = {}
        peak_count = read_feather_hm(f'../counting_csv_out/gencode_v29_tss/{f}', fix_id=True)
        #peak_count.loc[:, 'Target'] = t
        target_dict[t]['peak_count'] = peak_count
        print(t, target_dict[t]['peak_count'].shape)

    # %%
    peak_list = []
    for t in target_dict:
        peak_list.append(target_dict[t]['peak_count'])
    peaks_df = pd.concat(peak_list, axis=0)


    # %%
    peaks_df=sort_columns(peaks_df)
    peaks_df.shape
    # %%
    peaks_df = peaks_df.loc[filtered_df.Code, :]

    # %%
    col_sums = peaks_df.sum(axis=0)
    masked = col_sums[col_sums>3]

    # %%
    peaks_df = peaks_df.loc[:, masked.keys()]

    # %%
    for hm in target:
        hm_assay = filtered_df.query(f'`Target of assay` == "{hm}"').Code.tolist()
        peak_hm = peaks_df.loc[hm_assay,:]
        peak_hm = peak_hm.merge(filtered_df[['Code', 'Tissue']].set_index('Code'), left_index=True, right_index=True)
        tissue = list(peak_hm.Tissue.unique())
        print(hm, peak_hm.shape)
        for t in tissue:
            tissue_assay = peak_hm.query(f'Tissue =="{t}"').index.tolist()
            peaks_tissue = peak_hm.groupby('Tissue').get_group(t).iloc[:, :-1].astype('bool')
            checking = (peaks_tissue == 1).mean(0)
            change_cols = np.where(checking <0.3)
            print(peaks_df.shape[1]- len(change_cols[0]))
            peaks_df.loc[tissue_assay, np.take(peaks_df.columns.tolist(), change_cols[0])] = 0

    # %%
    obs=filtered_df.set_index('Code')
    var=list_domain_todf(peaks_df.columns.tolist())
    X=peaks_df.values
    peak_adata=anndata.AnnData(X, obs=obs, var=var, dtype='int32')

    # %%
    #### avg
    avg_sv = []
    for hm in target:
        hm_assay = filtered_df.query(f'`Target of assay` == "{hm}"').Code.tolist()
        peak_hm = peaks_df.loc[hm_assay,:].astype('bool').astype('int')
        peak_hm_filter = peak_hm.astype('bool').astype('int')
        data_path = rf'../data/blood_cell/{hm}/'
        signal = average_signalValue_pyranges(peak_hm, data_path, average=True)
        signal = signal*peak_hm_filter
        avg_sv.append(signal)

    # %%
    avg_sv = pd.concat(avg_sv, axis=0)
    print(avg_sv.shape, peaks_df.shape)
    avg_sv = avg_sv.loc[peaks_df.index, peaks_df.columns]

    # %%
    peak_adata.layers['count'] = peak_adata.X
    peak_adata.layers['average_signal'] = avg_sv

    # %%
    peak_adata.layers['rank_within_assay'] = rank_transformation(peak_adata.to_df('count'), scaled=None)

    # %%
    sc.set_figure_params(dpi=100, color_map = 'viridis_r')
    sc.settings.verbosity = 1
    sc.logging.print_header()

    # %%
    # compute clusters using the leiden method and store the results with the name `clusters`
    from matplotlib.pyplot import rc_context
    sc.pp.neighbors(peak_adata, n_neighbors=5, use_rep='X',n_pcs=10, metric='cosine', )
    sc.tl.leiden(peak_adata, key_added='clusters', resolution=0.5)
    sc.tl.umap(peak_adata, )
    with rc_context({'figure.figsize': (5, 5)}):
        sc.pl.umap(peak_adata, add_outline=True, 
                color=['Target of assay', 'Tissue', 'clusters'],
                legend_fontsize=12, legend_fontoutline=2,frameon=False,
                title='clustering of assay', palette='Set1')


    data = peak_adata.to_df('count')
    code = peak_adata.obs.sort_values(['Target of assay','Tissue'])
    data = data.loc[code.index, :]
    assay_anno= code['Target of assay'].astype('object')
    tissue_anno = code['Tissue'].astype('object')

    assay_pallete = dict(zip(assay_anno.unique(), 
                            sns.color_palette("husl", len(assay_anno.unique()))))

    assay_color = assay_anno.map(assay_pallete)

    tissue_pallete = dict(zip(tissue_anno.unique(), cc.glasbey))

    tissue_color = tissue_anno.map(tissue_pallete)

    row_colors = pd.DataFrame({'Histone':assay_color,
        'Tissue':[tissue_pallete[i] for i in tissue_anno],
                                })



    # cluster_map = sns.clustermap(data,
    #                             row_colors=row_colors,row_cluster=False, col_cluster=False)

    # hist = [mpatches.Patch(color=assay_pallete[name], label=name) for name in assay_pallete]
    # tis = [mpatches.Patch(color=tissue_pallete[name], label=name) for name in tissue_pallete]
    # plt.legend(handles=tis + hist, bbox_to_anchor=(0, 1),
    #         bbox_transform=plt.gcf().transFigure)
    # cluster_map.savefig(f'network_dataframe_majortissue/count_fig_{chrom}.png')
    # peak_adata.write_h5ad(f'network_dataframe_majortissue/adata_hm_tissue_{chrom}.h5ad', compression='gzip')

    # %%
    def filter_nonzero(data):
        col = data.sum(0)
        masked = col[col>0]
        df = data.loc[:,masked.keys()]
        return df

    # %%
    for hm in target:
        tissue = peak_adata.obs.query(f'`Target of assay` == "{hm}"')['Tissue'].unique()
        for t in tissue:
            if 'neutr' in t:
                continue
            subset = peak_adata[(peak_adata.obs['Target of assay'] == hm ) &
                                (peak_adata.obs['Tissue'] == t)]
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
            network._eigenvector_centrality(weight_name='weight', col_name='evcent_all')

            network._vertex_strength('weight')
            
            network.leidenPartition(weight_name='weight', name='leiden_modularity', partition_func=la.ModularityVertexPartition, use_abs_weight=True) 
            
            network._eigenvector_centrality(weight_name='weight', col_name='evcent_modularity', for_partition=True, partition_col='leiden_modularity')
            
            vertex_pearson = network.get_node_dataframe()
            vertex_pearson['degree'] = network._igraph.degree(loops=False)
            vertex_pearson['average_strength'] = vertex_pearson['strength']/vertex_pearson['degree']

            #save adj matrix as sparse matrix -> light 
            mat= scipy.sparse.csr_matrix(adjacency_matrix.values)
            with gzip.open(f'network_dataframe_majortissue/adjacency_matrix/{hm}_{t}_{chrom}_adjacency_matrix.mtx.gz', 'wb') as f:
                scipy.io.mmwrite(f, mat, precision=8)
            np.save(f'network_dataframe_majortissue/adjacency_matrix/{hm}_{t}_{chrom}_adjacency_matrix_region', adjacency_matrix.index.to_numpy())

            #save vertex_df
            vertex_pearson.reset_index().to_feather(f'network_dataframe_majortissue/vertex_df/{hm}_{t}_{chrom}_vertex.feather')
            print(f'Done {hm}-{t}-{chrom}')
    print('Finish', {chrom})

def main():
    chrom_list =[]
    for i in range(1, 23):
        chrom = f'chr{i}'
        chrom_list.append(chrom)
    workers = 5
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results =  executor.map(processing, chrom_list)
    
    for r in results:
        print(r)


if __name__ == '__main__':
    main()

