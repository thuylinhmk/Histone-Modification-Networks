# HISTONE MODIFICATION NETWORK - CORRELATION BASED NETWORK

Histone modifications are epigenetic marks that play a critical role in modifying gene expression and regulatory region function, with implications in various diseases. However, loci-loci interactions across large genomic contexts within- and between- chromosomes present a significant challenge in understanding the importance of a given histone modification within a given genomic context.

Network-based approach constructed within each chromosome can identify tissue-specified modules and high-rank loci overlaps with cellular regulatory functional terms.

1. hmEnv_req.txt: HPC conda environment requirements and packages
2. bedFile_reader.py: BEDFile reading class
3. corr_deepgraph.py: pairwise correlation calculation (later on update into GPU-acc)
4. graph_vis: Histone modification network construction code
