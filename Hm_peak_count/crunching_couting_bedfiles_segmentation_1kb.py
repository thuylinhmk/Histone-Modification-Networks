#!/bin/python

import os, sys, mmap
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import time
import gzip
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyranges as pr
import math
from functools import partial
from bedfile_reader import ReadBED


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

def loading_region(file_name: str, window_size: int=None) -> pr.PyRanges:
    def divide_region(region, window_size: int):
        start, end = region

        windows =[]
        for i in range(start, end + 1, window_size):
            window_start = i
            window_end = min(i + window_size - 1, end)
            windows.append((window_start, window_end))

        return windows

    regions = []
    with open(fr'{file_name}', 'r') as g:
        for line in g:
            chrom, start, end = line.strip().split(',')
            if window_size is None:
                regions.append([int(start), int(end)])
            else:
                regions += divide_region([int(start), int(end)], window_size)

    regions = list(set(map(tuple, regions)))
    regions = list(map(list, regions))
    regions_pr = pr.from_dict({'Chromosome': ['chr1']*len(regions),
                               'Start': [r[0] for r in regions], 
                               'End': [r[1] for r in regions]
                              })

    return regions_pr

def count_HM_in_sample(file_name, datapath, chrom, regions) -> list:
    """Region is a pyranges table with 3 columns: chromosome: str/cate, start:int, end:int"""
        
    sample_id = file_name.split('_')[1].split('.')[0]

    print(f'Processing {file_name}')
    #read bedfile
    sample = ReadBED(f'{datapath}/{file_name}')
    
    #get peak count number in window
    overlap = regions.count_overlaps(sample.get_peak_data())
    count = list(overlap.NumberOverlaps.to_numpy())
    row = [sample_id] + count
    return row

def parallel_process(workers, list_file, datapath, chrom, regions, save_name):
    #regions is pyranges
    
    #create fields
    FIELDS = ["Sample_ID",]
    regions = regions.assign('Name', lambda df: df.Chromosome.astype(str) + ":" + df.Start.astype(str) + "-" + df.End.astype(str))
    FIELDS = FIELDS + regions.Name.to_list()
    
    #call parallel processor
    with ProcessPoolExecutor(max_workers=workers) as executor:
        size=math.ceil(len(list_file) / workers)
        function = partial(count_HM_in_sample, datapath=datapath, chrom=chrom, regions=regions)
        doCount = executor.map(function, list_file, chunksize=size)

    # creating dataframe to save 
    countingList = [str(x).split('[')[1].split(']')[0] for x in doCount]
    towrite = [count.split(',') for count in countingList]

    df = pd.DataFrame(towrite, columns=FIELDS)
    #df['Sample_ID'] = df['Sample_ID']
    df = df.set_index('Sample_ID')
    df = df.astype('int')
    df = df.reset_index()

    print(f'write feather file{save_name}')
    df.to_feather(save_name)


def main():
    
    chrom ='chr1'
    start_time = 0
    target_list = ['H3K27ac', 'H3K27me3', 'H3K4me1', 'H3K9me3']
    #target_list = ["H3K27me3"]
    window_size=1000
    save_path = r"major_tissue/"
    n_core = mp.cpu_count()
    workers = mp.cpu_count()//2 - 1

    for hm in target_list:
        timer(0)

        datapath = fr'data/sorted_{hm}'
        bedfiles = os.listdir(datapath)

        print(f"Processing counting: {hm}, {chrom}, for files in {datapath}")

        #regions segmentation 1kb
        gene = loading_region(fr'region/{hm}_gene_region_majortissue.csv', window_size=window_size)
        reg = loading_region(fr'region/{hm}_regulatory_region_majortissue.csv', window_size=window_size)

        loop_regions = [gene, reg]
        for i in range(len(loop_regions)):
            if i == 0:
                save_name = fr"{save_path}{hm}/{hm}_1kb_gene_region_mt.feather"
                parallel_process(workers, 
                list_file=bedfiles, 
                datapath=datapath, 
                chrom=chrom, 
                regions=loop_regions[i], 
                save_name=save_name)
            elif i == 1:
                save_name = fr"{save_path}{hm}/{hm}_1kb_regulatory_region_mt.feather"
                parallel_process(workers, 
                list_file=bedfiles, 
                datapath=datapath, 
                chrom=chrom, 
                regions=loop_regions[i], 
                save_name=save_name)

        print(timer())

if __name__ == '__main__':
    main()
