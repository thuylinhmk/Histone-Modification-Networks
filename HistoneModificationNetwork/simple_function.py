import requests, json
import wget
import os
import shutil
import pandas as pd
import sys
import gzip
import anndata
import scanpy as sc
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy import stats
import numpy as np
import pyranges as pr
from bedfile_reader import ReadBED
import igraph as ig
import PyWGCNA
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import MinMaxScaler


DATA_TYPE_ENCODE = {'biosamples', 'experiments', 'files', 'analyses'}
FILE_TYPE = {"bed narrowPeak", "fastq", "all", 'tar'}
HEADERS = {'accept': 'application/json'}
ENCODE_URL = 'https://www.encodeproject.org/'
SOURCE_PATH = r"C:\Users\linhn\OneDrive\NguyenThuyLinh\Research BIOX7021"
DATA_DIRECTORY = r"C:\Users\linhn\OneDrive\NguyenThuyLinh\Research BIOX7021\data"

#searching with ENCODE ID
def search_ENCODE_ID(id_code, dataType):
    """Searche using sample name"""
    if dataType not in DATA_TYPE_ENCODE:
        raise ValueError("Only have one of following searching catagrory: %r." % DATA_TYPE_ENCODE)
    url = ENCODE_URL + dataType + '/' + id_code + '/?frame=object'
    return requests.get(url, headers=HEADERS).json()


#download files with encode id and file type
def download_files_ENCODE(
    id_code: str, 
    fileType: str, 
    output:str, #file name or directory
    assembly:str =None):

    """Download the file with experiment id and file type want to download"""

    #create this bar_progress method which is invoked automatically from wget
    def bar_progress(current, total, width=80):
        progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    genome_ref = None
    existed = os.listdir(output)
    if len(existed) != 0:
        existed = [x.split('.bed')[0] for x in existed]
    #check if available file type
    if fileType not in FILE_TYPE:
        raise ValueError("Only supporting one of following type: %r." % FILE_TYPE)

    experiment = search_ENCODE_ID(id_code, "experiments")
    count = 0
    for file in experiment['files']:
        fileName = file.split('/')[2]
        if fileName in existed:
            continue    
        search = search_ENCODE_ID(fileName, 'files')
        file_type = search['file_type']
        if fileType == "all" or file_type == fileType:
            if assembly is not None:
                genome_ref = search['assembly']
            if genome_ref == assembly or assembly is None:
                try:
                    if "preferred_default" in search.keys():
                        download_url = ENCODE_URL + file + "@@download"
                        wget.download(download_url, bar=bar_progress, out=output)
                        count += 1
                        print(file + "is downloaded")
                        print(f'Have download {count}')
                except:
                    print("there is something wrong with downloading" + file)
                    continue


#move multiple files to data directory file ending
def move_files_with_endings(source_directory, file_ending,
                            destination_directory, folder_name=None):
    sourcePath = fr"{source_directory}"
    sourceFiles = os.listdir(sourcePath)
    destinationPath = fr"{destination_directory}" 
    if folder_name != None:
        destinationPath += fr"/{folder_name}"
    for fileName in sourceFiles:
        if fileName.endswith(file_ending):
            shutil.move(os.path.join(sourcePath, fileName), os.path.join(destinationPath, fileName)) 

#load bed file
def read_bed_to_dataframe(
    filePath, compress_gzip=True):
    if compress_gzip == True:
        data_frame = pd.read_csv(filePath, sep='\t', comment='t', header=None, compression='gzip')
    else:
        data_frame = pd.read_csv(filePath, sep='\t', comment='t', header=Nones)
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak']
    data_frame.set_axis(header, axis=1, inplace=True)
    data_frame.sort_values(by=['chrom', 'chromStart'], inplace=True)
    return data_frame

#get bed file name
def bedFile_in_path(filePath):
    bedFile = []
    sourcePath = fr"{filePath}"
    sourceFiles = os.listdir(sourcePath)
    for file in sourceFiles:
        if file.endswith("bed.gz"):
            bedFile.append(file)
    return bedFile

#update and save to file
def save_samples_as_pickle(sample: dict):
    with open(sample, 'wb') as file:
        pickle.dump(sample, file)
        file.close()

def load_saved_sample_pickle(fileName: str):
    with open(fileName, 'rb') as file:
        mySample = pickle.load(file)
        file.close()
    return mySample

def read_annotation_gff3(fileName: str, feature: str, compression=True):
    file_ann = pd.read_table(fileName, compression='gzip', comment="#", sep = "\t", names = ['seqname', 'source', 'feature', 'start' , 'end', 'score', 'strand', 'frame', 'gene_info'])

    if feature == 'gene':
        genes_ann = file_ann[(file_ann.feature == feature)][['seqname', 'source', 'start', 'end', 'strand', 'gene_info']].copy().reset_index().drop('index', axis=1)

        gene_type = list(gm.split(';')[2].split('=')[1] for gm in genes_ann.gene_info)
        gene_name = list(gm.split(';')[3].split('=')[1] for gm in genes_ann.gene_info)
        gene_level = list(gm.split(';')[4].split('=')[1] for gm in genes_ann.gene_info) 

        genes_ann['gene_type'] = gene_type
        genes_ann['gene_name'] = gene_name
        genes_ann['gene_level'] = gene_level

        #sort by level and drop dup
        genes_ann = genes_ann.sort_values(by=['gene_level', 'seqname'], ascending=True).drop_duplicates('gene_name', keep='first').reset_index().drop('index', axis=1) 

    elif feature == 'exon':
        genes_ann = file_ann[(file_ann.feature == feature)][['seqname', 'source', 'start', 'end', 'strand', 'gene_info']].copy().reset_index().drop('index', axis=1)

        gene_type = list(gm.split(';')[4].split('=')[1] for gm in genes_ann.gene_info)
        gene_name = list(gm.split(';')[5].split('=')[1] for gm in genes_ann.gene_info)
        exon_number = list(gm.split(';')[8].split('=')[1] for gm in genes_ann.gene_info)
        gene_level = list(gm.split(';')[10].split('=')[1] for gm in genes_ann.gene_info)
        gene_strand = genes_ann['strand'].tolist()
        gene_exon_nb = list(f'{gene_name[i]}:exon {exon_number[i]}({gene_strand[i]})' for i in range(len(exon_number)))
    

        genes_ann['gene_type'] = gene_type
        genes_ann['gene_name'] = gene_exon_nb
        genes_ann['gene_level'] = gene_level
    
        #sort by level and drop dup
        genes_ann = genes_ann.sort_values(by=['gene_level', 'seqname'], ascending=True).drop_duplicates(['gene_name', 'strand'], keep='first').reset_index().drop('index', axis=1) 

    return genes_ann


def annotation_chr_region(region, feature: str, genome='hg38'): 
    #overlap check -> length

    if genome == 'hg38':
        ref = hg38_genes
    genes = []
    chrom_interest = ref[ref['seqname'] == region['chrom']]
    start = region['start']
    end = region['end']
    for row in chrom_interest.itertuples(index=False):
        chromStart = getattr(row, 'start')
        chromEnd = getattr(row, 'end')
        gene_name = getattr(row, 'gene_name')
        if start <= chromEnd and end >= chromStart:
            ol = _overlap(start, end, int(chromStart), int(chromEnd))
            if ol > 0:
                ann = fr"{gene_name}({ol})"
                genes.append(ann)
    
    if len(genes)>0:
            return ";".join(genes)
    else:
        return "NA"
    
def _overlap(s_str, s_end, ref_str, ref_end):
        length_overlap  = min(s_end, ref_end)-max(s_str, ref_str)
        return length_overlap

def _annotation_row(region, ref: pd.DataFrame): 
    genes = []
    chrom_interest = ref[ref['seqname'] == region['chrom']]
    start = region['start']
    end = region['end']
    for row in chrom_interest.itertuples(index=False):
        chromStart = getattr(row, 'start')
        chromEnd = getattr(row, 'end')
        gene_name = getattr(row, 'gene_name')
        if start <= chromEnd and end >= chromStart:
            ol = _overlap(start, end, int(chromStart), int(chromEnd))
            if ol > 0:
                ann = fr"{gene_name}({ol})"
                genes.append(ann)
    
    if len(genes)>0:
            return ";".join(genes)
    else:
        return "NA"

exon_ann = None
gene_ann = None

def annotation_region(
    data: pd.DataFrame, #should have chr, start, end
    feature: str, #feature want to annotation,
    annotation_file: str, #gff3 files, tsv file with '\t' delimiter,
    annotation_pd: pd.DataFrame = None # should have chr, start, end, gene_name
    ):
    global exon_ann
    global gene_ann

    if annotation_pd is None:
        if feature == 'exon' and exon_ann is None:
            exon_ann = read_annotation_gff3(annotation_file, feature)
        elif feature == 'gene' and gene_ann is None:
            gene_ann = read_annotation_gff3(annotation_file, feature)
        
        if feature == 'exon':
            data['gene_annotation'] = data.apply(
                _annotation_row, ref=exon_ann, axis=1)
        elif feature == "gene":
            data['gene_annotation'] = data.apply(
                _annotation_row, ref=gene_ann, axis=1)        
    else:
        data['gene_annotation'] = data.apply(
                _annotation_row, ref=annotation_pd, axis=1)  

    data['gene_name'] = list(x[0] for x in data['gene_annotation'].str.split('('))

def _create_tss_gene_df(
    genes_df: pd.DataFrame, #dataframe must have chr, start, end, strand, gene_name
    tss_df=pd.DataFrame, #df must have chr, start, end, strand, gene_name, peak_start
    ):

    #choosing avarage position of the same genes (1 gene have many TSS ID -> from different CAGE-ID)
    adverage_peak_start = tss_df.groupby('gene_name')['strand', 'peak_start'].mean().astype(int)

    #merge gene coordination and average tss cordination
    tss_gene = pd.merge(gene_ann[['seqname', 'start', 'end', 'strand', 'gene_name']], adverage_peak_start,  left_on='gene_name', right_index=True, copy=True)

    return tss_gene

def _create_regulation_domain(
        gene_tss_file: str, #gff4 file
        TSS_file= None, #path to tss coordination file with gene symbol anno \ with '\t' separation
        TSS_pd=None, #chr, start, end, strand, peak_start, gene_name
        extension_upstream= 5000, # tss -5000
        extension_downstream=10, # tss +10
        ):
        if TSS_file is not None and TSS_pd is None:
            tss_df = pd.read_csv(TSS_file, sep='\t', header=None, names=['chr', 'start', 'end', 'strand', 'peak_start', 'gene_name'])
        elif TSS_pd is not None and TSS_file is None:
            tss_df = TSS_pd
        else: 
            raise ArgumentError("please choose tss from file OR df")
        
        gene_ann = read_annotation_gff3(gene_tss_file, 'gene')
    
def STARsolo_to_anndata(
    path_folder: str, 
    cache: bool = True, 
    var: str = 'gene_symbols',
    cache_compression:str ='gzip') -> anndata.AnnData:

    print(f"loading data from {path_folder}")

    s = sc.read(
            f'{path_folder}/matrix.mtx.gz',
            cache=cache,
            cache_compression=cache_compression,
        ).T

    genes = pd.read_csv(f'{path_folder}/features.tsv.gz', header=None, compression='gzip', sep='\t')

    if var == 'gene_ids':
        s.var_names = genes[0].values
        s.var['gene_symbols'] = genes[1].values.astype('str')
        s.var['feature_type'] = genes[2].values.astype('str')
    elif var == 'gene_symbols':
        s.var_names = genes[1].values.astype('str')
        s.var_names_make_unique()
        s.var['gene_ids'] = genes[0].values.astype('str')
        s.var['feature_type'] = genes[2].values.astype('str')
    else:
        raise ArgumentError("var is not provided: choose 'gene_ids' or 'gene_symbols'")

    s.obs_names = pd.read_csv(f'{path_folder}/barcodes.tsv.gz', header=None, compression='gzip')[0]
    return s    


def filter_region(
        data:pd.DataFrame, 
        min_count: int=0, 
        min_sample: int=0, 
        inplace: bool=False):

        df = data.copy()
        col_sum = df.sum(axis=0)
        non_zero = df.astype(bool).sum(axis=0)

        if min_count:
            df = df.loc[:, col_sum >= min_count]
        
        if min_sample:
            df = df.loc[:, non_zero>=min_sample]
            
        if inplace:
            data = df.copy()
            return data
        else:
            return df

def read_corr_to_df(path: str):
    with open('corr_h3k27ac_high_gene_1kb.csv', 'r') as datafile:
        #numb = number_regions
        regions = []
        rows = []
        count=0
        for index, line in enumerate(datafile):
            data = line.strip().split(',')
            region = data[0]
            corr_values = data[1:]

            regions.append(region)
            if index == 0:
                rows.append(corr_values)
            else:
                corr = [None]*index + corr_values
                rows.append(corr)

    df = pd.DataFrame(rows, index=regions, columns=regions)
    return df


#HM counting analysis helper function
def highest_mean_expression(data: sc.AnnData, n_top:int):
    """Getting top mean count from anndata table of hm peak count"""
    #min count for each row
    data.var['mean_count'] = np.ravel(data.X.mean(0))
    #table with largest mean
    top_region = pd.DataFrame(data.var.nlargest(n_top, 'mean_count')['mean_count'])
    return top_region.index.to_list()

def extract_domain(data: pd.DataFrame, n_top_mean=int, n_top_variable=int) -> list:
    """extract n_top highest mean and n_top highly variable region. remove any overlap region"""
    #data need to be filtered
    ann_data = sc.AnnData(data)
    #top mean domain
    highest_mean = highest_mean_expression(ann_data, n_top_mean)
    
    #top variable domain 
    sc.pp.highly_variable_genes(ann_data, n_top_genes=n_top_variable)
    top_variable = list(ann_data[:,ann_data.var.highly_variable].to_df().columns)
    
    chosen_domain = list(set(highest_mean).union(set(top_variable)))
    return chosen_domain

def list_domain_todf(domains:list) -> pd.DataFrame:
    """From a list of domain string having format chr:start-end. create a pd.DataFrame from it with 3 columns: chrom, start, end and index as domain string """
    chrom = [r.split(':')[0] for r in domains]
    start = [int(r.split(':')[1].split('-')[0]) for r in domains]
    end = [int(r.split(':')[1].split('-')[1]) for r in domains]
    df = pd.DataFrame({'Chromosome':chrom, 'Start':start, 'End':end}, index=domains)
    df = df.sort_values(by=['Chromosome', 'Start'])
    return df

def write_domain_to_csv(domains: list, name: str) -> None:
    """write a csv file with list of domain strings -> 3 columns: chr, start, end"""
    df = list_domain_todf(domains)
    df.to_csv(name, header=False, index=False)
    
def sort_region_cols(data: pd.DataFrame):
    """As saving can cause region column to mixing up -> getting the string format of df and sort them accordingly chromosome"""
    regions = list(data.columns)
    chr1 = [c.split(':')[0] for c in regions]
    start = [c.split(':')[1].split('-')[0] for c in regions]
    end = [c.split(':')[1].split('-')[1] for c in regions]
    
    sort_df = pd.DataFrame({'chr': chr1, 'start': start, 'end': end})
    sort_df = sort_df.astype(dtype={'start':'int', 'end':'int'})
    sort_df = sort_df.sort_values(by='start').reset_index()
    sort_df = sort_df.drop(columns='index')
    
    sort_df['region_sorted'] = sort_df.agg('{0[chr]}:{0[start]}-{0[end]}'.format, axis=1)
    sort_region = list(sort_df['region_sorted'])
    sorted_df = data[sort_region]
    return sorted_df

def read_feather_hm(file_name: str, index_col="Sample_ID", fix_id:bool=False) -> pd.DataFrame:
    """Read feather file with index col = Sample_ID. some files during saving causing problem with extra quotes
       problems from crunching file into multiprocessor which return all outputs as string. will fix in future.
       -> set fix_id to True to fix it."""
    #index_col: name of the index column
    #file_type: type of the file. Using feather
    df = pd.read_feather(file_name)
    if fix_id:
        df['Sample_ID'] = [ID[1:-1] for ID in list(df['Sample_ID'])]
    df = df.set_index(index_col)
    final_df = sort_region_cols(df)
    return final_df

def list_merging_highly_correlated_close_regions(data: pd.DataFrame, threshold:int=0.9) -> list:
    """Grouping adjacent columns which have peak counting high |ce| from spearman's correlation
        threshold = 0.9"""
    #return col index number
    def distance_between_region(name_1:str, name_2:str):
        s1, e1 = name_1.split(':')[1].split('-')
        s2, e2 = name_2.split(':')[1].split('-')

        m1 = (int(s1) + int(e1))//2
        m2 = (int(s2) + int(e2))//2

        d = abs(m1 - m2)
        return d
    
    regions = list(data.columns)
    #find highly corr pairs -> columns index
    highly_corr_cols = []
    for i in range(data.shape[1]-1):
        rs, pvalue = stats.spearmanr(data.iloc[:, i], data.iloc[:, i+1])
        if abs(rs) >= threshold and pvalue < 0.05:
            if distance_between_region(regions[i], regions[i+1]) <= 1001:
                highly_corr_cols.append([i, i+1])
    
    #merging columns
    merge_list = []
    merge_cols = []
    for r in range(len(highly_corr_cols) - 1):
        if highly_corr_cols[r][1] == highly_corr_cols[r+1][0] and len(merge_cols)==0:
            merge_cols = list(dict.fromkeys(highly_corr_cols[r] + highly_corr_cols[r+1]))
            merge_cols.sort()
            continue
        elif len(merge_cols)!= 0 and merge_cols[-1] == highly_corr_cols[r+1][0]:
            merge_cols = merge_cols = list(dict.fromkeys(merge_cols + highly_corr_cols[r+1]))
            merge_cols.sort()
            continue
        else:
            if len(merge_cols)>0:
                merge_list.append(merge_cols)
                merge_cols = []  
    return merge_list

def merge_col_from_list(data: pd.DataFrame, merging: list) -> pd.DataFrame:
    """ Merge columns from df based on list
        list = [[columns merging together], []]
        value = sum(merged columns)
        """
    df = data.copy()
    col_name = list(df.columns)
    for cols in merging:
        col_sum = df.iloc[:, cols].sum(axis=1)
        #print(col_sum)
        new_col_name = col_name[cols[0]].split('-')[0] + '-' + col_name[cols[-1]].split('-')[1]
        #print(new_col_name)
        df[new_col_name] = col_sum
    
    merged_list = []
    for lst in merging:
        merged_list.extend(lst)
    to_drop = [col_name[x] for x in merged_list]
    df = df.drop(to_drop, axis=1)
    return df

def df_to_BED(file_name: str, data: pd.DataFrame, header=None) -> None:
    """write a bed file from pd.dataframe"""
    if data.shape[1] < 3:
        raise ValueError("the dataframe should have at least 3 columns: chrom, start, end")
    f = open(file_name, "w")
    if header:
        f.write(header + '\n')
    rows = data.values.tolist()
    if data.shape[1] == 3:
        for row in rows:
            f.write("%s\t%d\t%d" % (row[0], row[1], row[2]))
            f.write('\n')
    f.close()
    
def write_feather_df(data: pd.DataFrame, name: str) -> None:
    """Write feather file from df with name"""
    df = data.reset_index()
    df.to_feather(name)
    print(f"Done saving dataframe as: {name}")


def average_signalValue_pyranges(data: pd.DataFrame, bedfile_path: str, average=True, prefix:str = None) -> pd.DataFrame:
       #data: pd.DataFrame with region and peak count from each sample, sorted columns accrodding to their position of chr1
    #bedfiles_path: path to the sorted bedfile of sample
    count = 1
    
    #creating signalValue dataframe
    #signal_df = data.copy()
    signal_list = []
    if prefix:
        #file list in path
        hm_files = [f"{prefix}{x}.bed.gz" for x in data.index.tolist()]
    else:
        hm_files = [f"{x}.bed.gz" for x in data.index.tolist()]
    
    #string represent of region, region df [chrom, start, end]
    region_df = list_domain_todf(list(data.columns))
    region_df.columns = ['Chromosome', 'Start', 'End']
    region_df = region_df.reset_index().rename({'index': 'Name'}, axis=1)
    region_pr = pr.PyRanges(region_df)    
    
    # getting signalValue
    # sorted file -> chr1 appears first - not necessary  
    for file in hm_files:
        #print(1, end='\r', flush=True)
        #sample id
        if prefix: 
            name = file.split(prefix)[1].split('.bed')[0]
        else:
            name = file.split('.bed')[0]
        
        #read sample in pyranges. get peak positions table
        sample = ReadBED(f"{bedfile_path}/{file}")
        peaks = sample.get_peak_data()
        
        #print(2, end='\r', flush=True)
        #extract peak overlap with which region: intersect() -> merge with peak table to get annotate region into matching peak
        peak_overlap_region = region_pr.join(peaks, how='left').df
        peak_overlap_region['signalValue'] = peak_overlap_region['signalValue'].replace(-1, 0)
        if average:
            signal_avg = peak_overlap_region[['Name', 'signalValue']].groupby('Name').mean()
        else:
            signal_avg = peak_overlap_region[['Name', 'signalValue']].groupby('Name').sum()
         
        avg = region_pr.df.merge(signal_avg, left_on='Name', right_on='Name')
        row = [name] + avg.signalValue.tolist()
        signal_list.append(row)        
      
        #print(6, end='\r', flush=True)
        print(f"done with {count} files over {len(hm_files)}", end='\r')
        count += 1 
        
    #print(7, end='\r', flush=True)  
    #create dataframe
    fields = ['Sample_ID'] + list(region_pr.Name)
    signal_df = pd.DataFrame(signal_list, columns = fields)
    signal_df = signal_df.set_index('Sample_ID')
  
    print(f"Done processing", end="\r")                
    return signal_df

def merging_high_corr_col(data):
    merge_cols = list_merging_highly_correlated_close_regions(data)
    merged_df = merge_col_from_list(data, merge_cols)
    merged_df = sort_region_cols(merged_df)
    return merged_df

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

def get_adjacency_matrix(data, threshold, absolute=True, up_threshold= None, keep_diag=True):
    """ Return adjacency matrix based on input threshold
        Args:
            - Data: data
            - absolute (boolean): If you want to use absolute values for data (Default = True)
            - threshold: Threshold for the adjacency. Any value below this threshold will be change into 0
            - up_threshold: if you want to eliminate upper value        
    """
    df = data.copy()
    if absolute:
        df[abs(df) < threshold] = 0
        if up_threshold is not None:
            df[abs(df) > up_threshold] = 0
    else:
        df[df < threshold] = 0 
        if up_threshold is not None:
            df[df > up_threshold] = 0
        

    np_df = df.values
    if keep_diag == False:
        np.fill_diagonal(np_df, 0)
    else: 
        np.fill_diagonal(np_df, 1)
    df = pd.DataFrame(np_df, index=df.index, columns=df.columns)    
    return df

def igraph_construct(adj_data, metadata, meta_col=None, meta_name=None, self_loop=False):
    np_df = adj_data.to_numpy()
    if not self_loop:
        np.fill_diagonal(np_df, 0)
    network=ig.Graph.Adjacency((abs(np_df)>0).tolist())
    network.es['weight'] = np_df[np_df.nonzero()]
    network.vs['name'] = adj_data.index.tolist()
    if meta_col is None and meta_name is None:
        meta_col = metadata.columns.tolist()
        meta_name= metadata.columns.tolist()
    elif meta_col is None or meta_name is None:
        raise ValueError('have to implement both meta_col and meta_name or leave both of them None')

    for index, col in enumerate(meta_col):
        network.vs[meta_name[index]] = metadata.loc[network.vs['name'], col]    
    return network

def pyranges_from_vertex(vertex_df):
    region_pr = pr.PyRanges(list_domain_todf(vertex_df.index.tolist()))
    region_pr.name = region_pr.Chromosome.astype('str') + ':' + region_pr.Start.astype('str') + '-' + region_pr.End.astype('str')
    return region_pr

def feature_annotation(region_pr, anno_pr):
    anna = region_pr.join(anno_pr,apply_strand_suffix=False)[['Chromosome', 'Start', 'End', 'name', 'Feature', 'gene_name', 'Strand']].df

    anna = anna.sort_values(['Start','Feature']).drop_duplicates(subset=['name', 'gene_name'], keep='first')

    anna['gene_symbol'] = anna.groupby(['name'])['gene_name'].transform(lambda x: ', '.join(x))

    anna =  anna.sort_values(['Start','Feature']).drop_duplicates(subset=['name'], keep='first')

    anna.drop(['gene_name', 'Strand'], inplace=True, axis=1)
    anna.reset_index(drop=True, inplace=True)
    anna.set_index('name', inplace=True)
    return anna

def sort_columns(data):
    x = pr.PyRanges(list_domain_todf(data.columns.tolist()).reset_index())
    df = data[x.index.tolist()]
    return df