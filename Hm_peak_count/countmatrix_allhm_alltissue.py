import numpy as np
import pandas as pd
import os
from simple_function import *
from graph_vis import *
import anndata

chrom = ['chr1']
target = ['H3K27me3', 'H3K4me3', 'H3K4me1', 'H3K27ac', 'H3K9me3']

count_files=os.listdir('../counting_csv_out/gencode_v29_tss/')
count_files= [x for x in count_files if f'{chrom[0]}_' in x]

assay_info = []
for t in target:
    file_info = pd.read_csv(f'../Histone_data_tsv/{t}_file_info.csv')
    file_info.columns = ['Code', 'Accession', 'Target of assay', 'Tissue', 'Biosample summary']
    assay_info.append(file_info)

assays = pd.concat(assay_info, axis=0)

target_dict = {}
for f in count_files:
    t = f.split('_')[0]
    target_dict[t] = {}
    peak_count = read_feather_hm(f'../counting_csv_out/gencode_v29_tss/{f}', fix_id=True)
    #peak_count.loc[:, 'Target'] = t
    target_dict[t]['peak_count'] = peak_count
    print(t, target_dict[t]['peak_count'].shape)

peak_list = []
for t in target_dict:
    peak_list.append(target_dict[t]['peak_count'])
peaks_df = pd.concat(peak_list, axis=0)

assays = assays.sort_values(['Target of assay', 'Tissue'], ignore_index=True)

peaks_df=sort_columns(peaks_df)
print(f'assays x regions: {peaks_df.shape}')


#tissue with number of assays >=10
filters_assay = assays.groupby(['Target of assay', 'Tissue']).size()
filters_assay = filters_assay[filters_assay>=10].reset_index().iloc[:, :2]
filters_assay_tuples = list(filters_assay.itertuples(index=False, name=None))
filtered_df = assays[assays[['Target of assay', 'Tissue']].apply(tuple, axis=1).isin(filters_assay_tuples)]
peaks_df = peaks_df.loc[filtered_df.Code, :]
print(f'number of assay in tissue with min assasy >=10: {peaks_df.shape}')

target = list(assays['Target of assay'].unique())
print(f'Histone: {target}')

for hm in target:
    tissue = list(filtered_df.query(f'`Target of assay` == "{hm}"').Tissue.unique())
    for t in tissue:
        code = filtered_df.query(f'`Target of assay` == "{hm}" and Tissue == "{t}"').Code.to_list()
        cols = peaks_df.loc[code, :].astype('bool').sum()
        masked = cols / len(peaks_df.loc[code, :])
        cols = cols[masked<0.5]
        peaks_df.loc[code, cols] = 0

print("Saving files")
peaks_df.reset_index().to_feather('filtered_assays_region.feather')
filtered_df.to_csv('filtered_assay.csv', index=False)
print("Done!")