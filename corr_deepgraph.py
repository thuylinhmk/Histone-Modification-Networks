# data i/o
import os

# compute in parallel
from multiprocessing import Pool

# the usual
import numpy as np
import pandas as pd

import deepgraph as dg

#timing
import time

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

print('read data file')
h3k27ac_gene = pd.read_feather('../processing_data/H3K27ac/cluster/h3k27ac_gene_chr1_6cluster.feather')
h3k27ac_gene = h3k27ac_gene.set_index('index')

cluster = h3k27ac_gene.pop('Cluster')

X = h3k27ac_gene.to_numpy()

n_features = X.shape[0]
n_samples = X.shape[1]

# uncomment the next line to compute ranked variables for Spearman's correlation coefficients
X = X.argsort(axis=1).argsort(axis=1)

# whiten variables for fast parallel computation later on
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# save in binary format
print('save tmp ranked matrix')
np.save('tmp/ranked', X)
np.save('tmp/region_name', h3k27ac_gene.index.to_numpy())
np.save('tmp/cluster_np', cluster.to_numpy())
# parameters (change these to control RAM usage)
step_size = 32000
n_processes = 20


# load samples as memory-map
X = np.load('tmp/ranked.npy', mmap_mode='r')

# create node table that stores references to the mem-mapped samples
v = pd.DataFrame({'index': range(X.shape[0])})

# connector function to compute pairwise pearson correlations
def corr(index_s, index_t):
    features_s = X[index_s]
    features_t = X[index_t]
    corr = np.einsum('ij,ij->i', features_s, features_t) / n_samples
    return corr

# index array for parallelization
pos_array = np.array(np.linspace(0, n_features*(n_features-1)//2, n_processes), dtype=int)

# parallel computation
def create_ei(i):

    from_pos = pos_array[i]
    to_pos = pos_array[i+1]

    # initiate DeepGraph
    g = dg.DeepGraph(v)

    # create edges
    g.create_edges(connectors=corr, step_size=step_size,
                   from_pos=from_pos, to_pos=to_pos)
    edges_values = g.e
    to_save = edges_values.reset_index()
    # store edge table - will remove late for saving space
    to_save.to_feather('correlations/{}.feather'.format(str(i).zfill(3)))

# computation
if __name__ == '__main__':
    timer(0)

    print('calculating......', end='\r')
    os.makedirs("correlations", exist_ok=True)
    indices = np.arange(0, n_processes - 1)
    p = Pool(n_processes)
    for _ in p.imap_unordered(create_ei, indices):
        print(f'working hard......', end='\r')
    
    p.close()
    p.join()

    print("Done. Writing correlation file......")
    files = os.listdir('correlations')
    files.sort()
    firstFile = True
    for f in files:
        path = fr'correlations/{f}'
        if first_file:
            corr = pd.read_feather(path)
            corr.columns = ['node1', 'node2', 'rsh']
            corr['arsh'] = abs(corr['rsh'])
            corr = corr.query('arsh > 0.7')
            first_file = False
        else:
            read_file = pd.read_feather(path)
            read_file.columns = ['node1', 'node2', 'rsh']
            read_file['arsh'] = abs(read_file['rsh'])
            read_file = read_file.query('arsh > 0.7')
            corr = pd.concat([corr, read_file])
        try:
            os.rmdir(path)
            print("tmp file directory is deleted to save space")
        except OSError as error:
            print("Error occured: %s : %s" % (path, error.strerror))
    
    corr.reset_index(drop=True, )
    region_name = dict(enumerate(h3k27ac_gene.index.to_numpy()))

    corr['node1'] = corr['node1'].map(region_name)
    corr['node2'] = corr['node2'].map(region_name)
    corr.to_feather('spearman_corr_h3k27ac_gene.feather')

    print(f'All done. correlated is saveed in file name: spearman_coprr_h3k27ac_gene.feather')
    print(timer())