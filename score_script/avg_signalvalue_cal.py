import pandas as pd
import numpy as np
from simple_function import *
import scanpy as sc
import os
import gzip
from bedfile_reader import ReadBED
import pyranges as pr
from scipy import stats
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

major_tissue = ['heart right ventricle', 'heart left ventricle', 'spleen', 'sigmoid colon','stomach']
h3k27ac = pd.read_csv('../Histone_data_tsv/H3K27ac_files_info.csv')
h3k27me3 = pd.read_csv('../Histone_data_tsv/H3K27me3_tissue_info.csv')
h3k4me1 = pd.read_csv('../Histone_data_tsv/H3K4me1_file_info.csv')
h3k9me3 = pd.read_csv('../Histone_data_tsv/H3K9me3_file_info.csv')
filter_file = {}
filter_file['H3K27ac'] = h3k27ac.query(f'Tissue in {major_tissue}').Code.tolist()
filter_file['H3K27me3'] = h3k27me3.query(f'Tissue in {major_tissue}').Code.tolist()
filter_file['H3K4me1'] = h3k4me1.query(f'Tissue in {major_tissue}').Code.tolist()
filter_file['H3K9me3'] = h3k9me3.query(f'Tissue in {major_tissue}').Code.tolist()

def timer(t=None):
    global start_time
    if t is not None:
        start_time = time.time()
        return
    else:
        t1 = time.time()
        t = t1 - start_time
        start_time = t1
        hours, rem = divmod(t, 3600)
        minutes, seconds = divmod(rem, 60)
        return("Ellapsed time {:0>2}:{:0>2}:{:06.3f}".format(int(hours),int(minutes),seconds))

def avg_sv(region_pr, name, sorted_path, data):
    #sample id
    file = f'sorted_{name}.bed.gz'
    #read sample in pyranges. get peak positions table
    sample = ReadBED(f"{sorted_path}{file}")
    peaks = sample.get_peak_data()
        
    print('overlap', end='\r', flush=True)
    #extract peak overlap with which region: intersect() -> merge with peak table to get annotate region into matching peak
    peak_overlap = region_pr.intersect(peaks).df
    peak_match = peaks.df.merge(
        peak_overlap[['Chromosome', 'Start', 'End', 'Name']], on=['Chromosome', 'Start', 'End'])  
        
    print('signal sum', end='\r', flush=True)
    #calculate sum of signal value
    sum_signal_value = peak_match[['Chromosome', 'signalValue', 'Name']].groupby(['Name']).sum()
        
    print('avg', end='\r', flush=True)
    #match signal to regions
    signal_to_region = pd.merge(region_pr.df, sum_signal_value, how='left', on="Name")

    signal_to_region['signalValue'] = signal_to_region['signalValue'].fillna(value=0)
    #print(signal_to_region, flush=True)
        
    print(5, end='\r', flush=True)
    #replace sum signalValue 
    signal = signal_to_region['signalValue'].to_numpy()
    avg = signal / data.loc[name, :]
    avg = avg.fillna(value=0)
    row = [name] + list(avg)   
    return row 


def average_signalValue_pyranges(hm: str):
    print(f'start calculating for {hm}')
    timer(0)
    #data: pd.DataFrame with region and peak count from each sample, sorted columns accrodding to their position of chr1
    #bedfiles_path: path to the sorted bedfile of sample
    count = 1

    sorted_path = fr'../data/sorted_{hm}/'
    gene_merged = read_feather_hm(f'../major_tissue/{hm}/{hm}_peakcount_merged_gene.feather')
    reg_merged = read_feather_hm(f'../major_tissue/{hm}/{hm}_peakcount_merged_reg.feather')
    
    #creating signalValue dataframe
    #signal_df = data.copy()
    signal_list_gene = []
    signal_list_reg = []   
    
    #print('file in path')
    #file list in path
    hm_files = filter_file[hm]
    #print(hm_files)

    print('load region')
    #string represent of region, region df [chrom, start, end]
    region_gene = list_domain_todf(list(gene_merged.columns))
    region_gene.columns = ['Chromosome', 'Start', 'End']
    region_gene_pr = pr.PyRanges(region_gene)
    region_gene_pr.__setattr__('Name', list(gene_merged.columns))

    region_reg = list_domain_todf(list(reg_merged.columns))
    region_reg.columns = ['Chromosome', 'Start', 'End']
    region_reg_pr = pr.PyRanges(region_reg)
    region_reg_pr.__setattr__('Name', list(reg_merged.columns))

    print('getting signalValue') 
    for file in hm_files:
        print(f'loop file 1: {hm}')
        gene_sv_cal = avg_sv(region_gene_pr, file, sorted_path, gene_merged)
        signal_list_gene.append(gene_sv_cal)

        reg_sv_cal = avg_sv(region_reg_pr, file, sorted_path, reg_merged)
        signal_list_reg.append(reg_sv_cal)        
        
        print(f'loop file 2 {hm}')
        print(f"done with {count} files over {len(hm_files)}", end='\r')
        count += 1 
        
    print(f'Write file for {hm}')
    #create dataframe
    gene_col = ['Assay_ID'] + gene_merged.columns.tolist()
    signal_gene = pd.DataFrame(signal_list_gene, columns = gene_col)
    signal_gene.to_feather(f'../major_tissue/{hm}/{hm}_avg_signal_gene_majortissue.feather')
    reg_col = ['Assay_ID'] + reg_merged.columns.tolist()
    signal_reg = pd.DataFrame(signal_list_reg, columns=reg_col)
    signal_reg.to_feather(f'../major_tissue/{hm}/{hm}_avg_signal_reg_majortissue.feather')
  
    print(f"Done processing {hm}:", timer())                


def main():
    target_list = ['H3K27ac', 'H3K27me3', 'H3K4me1', 'H3K9me3']
    workers = 6
    h3k27ac = pd.read_csv('../Histone_data_tsv/H3K27ac_files_info.csv')
    h3k27me3 = pd.read_csv('../Histone_data_tsv/H3K27me3_tissue_info.csv')
    h3k4me1 = pd.read_csv('../Histone_data_tsv/H3K4me1_file_info.csv')
    h3k9me3 = pd.read_csv('../Histone_data_tsv/H3K9me3_file_info.csv')
    hm_list = ['H3K27ac', 'H3K27me3', 'H3K4me1', 'H3K9me3']
    
    filter_file = {}
    filter_file['H3K27ac'] = h3k27ac.query(f'Tissue in {major_tissue}').Code.tolist()
    filter_file['H3K27me3'] = h3k27me3.query(f'Tissue in {major_tissue}').Code.tolist()
    filter_file['H3K4me1'] = h3k4me1.query(f'Tissue in {major_tissue}').Code.tolist()
    filter_file['H3K9me3'] = h3k9me3.query(f'Tissue in {major_tissue}').Code.tolist()
    
    #call parallel processor
    with ProcessPoolExecutor(max_workers=workers) as executor:
        job = executor.map(average_signalValue_pyranges, target_list)
    for result in job:
        print(result)

if __name__ == '__main__':
    main()
