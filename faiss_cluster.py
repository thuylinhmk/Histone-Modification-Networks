import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from simple_function import * #need to divide the function in this into small specific .py file
from typing import Union


class FaissCluster:
    """K-means cluster using faiss"""
    def __init__(self, data: pd.DataFrame, gpu: bool=False):
        """
            Arguments:
                data (pd.DataFrame):
                    - rows are features (region) and columns are observations (ChIP-seq assay).
                    - values inside will be transform to 'float32' as faiss is more compatible with float32 type
                
                n_cluster (int):
                    - number of clusters 
                gpu (bool):
                    - True: if you want to using gpu version. <Not supported rightnow>
                    - False: do not use GPU
        """
        self.data = data
        self.data_np = self.data.to_numpy().astype(np.float32)
        self.n_features = data.shape[0]
        self.n_obs = data.shape[1]
        self.gpu = gpu
        self.regions_info = list_domain_todf(self.data.index.to_list())
        self.assays_info = pd.DataFrame(index = self.data.columns)
        self.cluster = None
        self.scaled_np = None
        self.scaled_df = None
        self.cluster_table = None

    
    def get_data(self):
        """ Show copy version of current data """
        return self.data.copy()
    
    def get_region_info(self):
        return self.regions_info.copy()
    
    def get_assay_info(self):
        return self.assays_info.copy()
    
    def get_scaled_data(self) -> pd.DataFrame:
        return self.scaled_df.copy()
    
    def scale_data(self, method: str='MinMax', inplace: bool=False) -> None:
        """ Scale the data using sklearn MinMaxScaler
            Arguments:
                method (string):
                    - 'MinMax': call MinMaxScaler() on data
                inplace (bool): 
                    - True: if scale on the orginal data
                    - False: create a copy for scale df
        """
        if method == 'MinMax':
            scaler = MinMaxScaler()
            self.scaled_np = scaler.fit_transform(self.data_np)
            self.scaled_df = pd.DataFrame(self.scaled_np, 
                                    columns = self.data.columns, 
                                    index = self.data.index)
        else: 
            raise ValueError(f'Not suported {method} method')

        if inplace:
            self.data = self.scaled_df
            self.data_np = self.data.to_numpy()
        

    def Kmeans(self, n_cluster: int, 
               OG_data: bool=False, niter: int=50) -> None:
        """ Create and train faiss.kmeans

            Arguments:
                n_cluster (int):
                    - number of clusters you want to train
                OG_data (boolean):
                    - True: if you want to train on the original data
                    - False: train on your scaled data. 
                niter (int):
                    - number of iteration
                    - default: 50
                verbose (bool):
                    - using verbose mode
                    - default: True
            Default: 
                - Default: train on scaled data. If your data is not scaled (self.scaled_np == None). Train on orginal data
            Return: None
            
        """

        #create kmeans cluster with n_obs and n_cluster
        self.n_cluster = n_cluster

        self.cluster = faiss.Kmeans(self.n_obs, self.n_cluster, gpu=self.gpu, niter=niter, verbose=True)
        

        #train cluster and set up cluster table
        if OG_data or self.scaled_np is None:
            self.cluster.train(self.data_np)
            self._assign_cluster(self.data_np)
            self._set_up_region_query(self.data_np)
            self.add_info(self.cluster_table, 'regions')     
        else:
            self.cluster.train(self.scaled_np)
            self._assign_cluster(self.scaled_np)
            self._set_up_region_query(self.scaled_np)
            self.add_info(self.cluster_table, 'regions') 


    def _assign_cluster(self, train_data) -> None:
        """ Assign region with cluster """
        if not self.cluster:
            raise AttributeError("You dont have a cluster training!")
    
        # assign cluster
        _, I = self.cluster.index.search(train_data, 1)
        self.cluster_table = pd.DataFrame(I, 
                        index = self.data.index, 
                        columns = ['Cluster'])

    def get_cluster_table(self) -> pd.DataFrame:
        return self.cluster_table.copy()  
    
    def add_info(self, new_info: pd.DataFrame, 
                into: Union["regions", "assays"]) -> None:
        """ Add additional information about regions or assay
            Arguments:
                new_info (pd.DataFrame):
                    - Dataframe contains new info to add.
                    - need to have same index as object left or right index
        """        
        if into == "regions":
            self.regions_info = pd.merge(self.regions_info, new_info, 
                    left_index = True, right_index= True)
        elif into == "assays":
            self.assays_info = pd.merge(self.assays_info, new_info,
                    left_index = True, right_index = True)

    def _set_up_region_query(self, train_data):
        quantizer = faiss.IndexFlatL2(self.n_obs)
        self.search_index = faiss.IndexIVFFlat(quantizer, 
                                               self.n_obs, 
                                              self.n_cluster,
                                              faiss.METRIC_L2)
        self.search_index.train(train_data)
        self.search_index.add(train_data)
        
    def _query_closest_vector(self, query, n_closest: int):
        """ Search n_closest data vectors to your query(ies)
            Arguments:
                query: 
                    - Query vector(s): need to have the same length as n_obs
                n_closest:
                    - Number of vectors you want to return
            Return:
                distance and index of data vectors 
        """

        D, I = self.search_index.search(query, n_closest)

        return D, I
    
    def _query_closest_cluster(self, query, n_closest: int):
        """ Search for the n_closest cluster to your query(ies)
            Arguments:
                query: 
                    - Query vector(s): need to have the same length as n_obs
                n_closest:
                    - Number of vectors you want to return
            Return:
                distance and index of data vectors
        """
        D, I = self.cluster.index.search(query, n_closest)
        return D, I

    def view_cluster_members(self, cluster_numb: int, OG_Data: bool=False):
        region_index = list(self.get_cluster_table().query(f'Cluster == {cluster_numb}').index)
        if OG_Data or (not self.scaled_np):
            cluster = self.data.loc[region_index,]
        else:
            cluster = self.scaled_df.loc[region_index,]
        return cluster
        
    def get_top_clostest_centrois(self, n_top: int, 
                                  cluster: Union[int, str, list[int]]='all',
                                  OG_data: bool=False):
        """ Getting top region closest to the cluster centroids """
        if cluster == 'all':
            query = self.cluster.centroids
        elif isinstance(cluster, int) and cluster < len(self.n_cluster):
            query = self.cluster.centrois[cluster]
        else:
            query = [self.cluster.centrois[i] for i in cluster]

        _, I = self._query_closest_vector(query, n_top)

        #set up table to return
        if OG_data:
            df = self.data.iloc[I.flatten(),]
        else:
            df = self.scaled_df.iloc[I.flatten(),]
        
        df = pd.merge(self.cluster_table, df, 
                        left_index=True, right_index=True)
        return df
        
    
