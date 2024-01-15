import networkx as nx
import pandas as pd
import numpy as np
import igraph as ig
import leidenalg as la
import pyarrow as pa
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial as sp
import plotly.graph_objects as go
from matplotlib.pyplot import gcf
import matplotlib.colors as mcolors

PALETTE=sns.color_palette('tab20').as_hex() #20 colors

class HMNetwork:
    """ Create Network from correlation of HM regions """
    def __init__(self, name: str=None):
        """
            Arguments: 
                weight (pd.DataFrame): 
                    -dataframe with at least 3 columns -> node1, node2, weight
                    -HM: pairwise correlation
                meta_data (pd.DataFrame):
                    -dataframe with information about each node
                    -index = name of node
                    -default: None (option and can add more later)
        """
        self._adjacency_matrix = None
        self._meta_data = None
        self._igraph = None
        self._vertex_df= None
        self._edge_df = None
        self.directed=None
        self.partitioned=False

        if name:
            self.name=name
        else:
            self.name=None

    def from_edge_dataframe(self, 
                       edge_df: pd.DataFrame,
                       metadata: pd.DataFrame = None,
                       directed: bool = False):
        """
            Construct graph from edge dataframe
            Arguments: 
                - edge_df: dataframe contains edge data with col1 and col2 as source and target. Other columns will be added as edge attributes
                - metadata: first columns must be unique name of vertices, other columns will be added as vertex attrs)

        """
        self._igraph = ig.Graph.DataFrame(edge_df, 
                                        directed=directed,
                                        vertices=metadata,
                                        use_vids=False)
        self.directed=directed
        self._set_up_attr_df()

    def from_adjacency(self, 
                       adjacency_matrix,
                       directed: bool = False,
                       loops:bool = False):
        """
            Contruct graph from adjacency matrix
            - adjacency_matrix: 
                pd.DataFrame: vertex name will be auto assigned as df index, values in the dataframe will be assigned of edge attr 'weight'
                numpy.array: vertex number id will be assign from 0, 
                values will be added as edge attr 'weight'
            - directed: boolean
                (Default: False): undirected graph
                True: directed graph
        """
        if directed:
            self._igraph = ig.Graph.Weighted_Adjacency(adjacency_matrix,
                                            mode='directed', loops=loops)
        else:
            self._igraph = ig.Graph.Weighted_Adjacency(adjacency_matrix,
                                            mode='upper', loops=loops)
        self.directed=directed
        self._set_up_attr_df()
    
    def _set_up_attr_df(self):
        self._vertex_df = self._igraph.get_vertex_dataframe()
        self._edge_df = self._igraph.get_edge_dataframe()
    
    def add_nodes_attr(self, data: pd.DataFrame, use_vids=False) -> None:
        """ Add node attribute.
            Arguments: 
                data (pd.DataFrame): pandas dataframe containing attributes' values with index are name of vertices (using 'name' of vertices) or vertex IDs (int from 0)
                - use_vids: 
                    True if using verterx IDs (0-...)
                    (Default) False if using name attribute of vertex
                    
        """
        name_attributes = data.columns.tolist()

        if use_vids:
            data = data.sort_index()
        else:
            index = self._igraph.vs['name']
            data = data.loc[index,:]
        
        for attr in name_attributes:
            self._igraph.vs[attr] = data.loc[:, attr].tolist()

    
    def leidenPartition(self, 
                        weight_name: str,
                        partition_func=la.ModularityVertexPartition,
                        niter: int=-1, 
                        seed:int=218,
                        name:str ='leiden_membership', 
                        use_abs_weight=False,
                        add_membership=True,
                        **args) -> None:
        """ Using Leiden to optimize graph
            Arguments: 
                weight_name: str
                    - name of weight column to use
                partition:
                    - Default: using la.ModularityVertexPartition
                    - Supported partition algorithms of leidenalg package
                name: str
                    - name to use when add leiden membership as vertex attr
                use_abs_weight: bool
                    - Default: False - use the original weight values
                    - True: use absolute value of weight values
                add_membership: bool
                    - Default: True - add membership as a new attr in vertices
                    - False: Do not add memebership in vertices and return partition instead
                niter (int):
                    - Number of iteration to train for partition
                    - Default: -1 -> train until no more changing
                seed (int):
                    - random seeding number
        """
        if use_abs_weight:
            self._igraph.es[f'abs_{weight_name}'] = list(map(abs, 
                                                self._igraph.es[weight_name]))
            partition = la.find_partition(self._igraph, 
                                        partition_func,
                                        n_iterations=niter, 
                                        weights=f'abs_{weight_name}',
                                        seed=seed,
                                        **args)
        else:
            try:
                partition = la.find_partition(self._igraph, 
                                        partition_func,
                                        n_iterations=niter, 
                                        weights=weight_name,
                                        seed=seed,
                                        **args)
            except BaseException as e:
                raise BaseException(f'{e} Please try setting use_abs_weight=True')
        
        print(f'Number of modules: {len(np.unique(partition.membership))}')
        sig = la.SignificanceVertexPartition.FromPartition(partition).quality()
        print(f'Paritition quality: {sig}')
        
        if add_membership:
            #annote module in igraph
            mem = partition.membership
            self._igraph.vs[name] = mem
        else:
            return partition
    
    def add_vertex_partition(self,
                             membership,
                             attr:str, 
                             col_name:str = None):
        """
            Add membership/cluster/partition columns
                membership: list or pd.DataFrame
                    list with membership order as vertex order in igraph
                    pd.DataFrame: index column with name of vertices
                attr: str name to add into vertex df
                col_name: if pd.DataFrame, name of col containing info about membership
        """
        if isinstance(membership, pd.DataFrame):
            if col_name is None:
                raise ValueError('col_name cannot be None with membership as pandas dataframe')
            else:
                self._igraph.vs[attr] = membership.loc[self._igraph.vs['name'], col_name]
        elif isinstance(membership, list):
            self._igraph.vs[attr] = membership
        else:
            raise ValueError("membership has to be list or pandas dataframe")
        
    def _create_node_color(self, graph, color_by: str, palette):
        """ Create color list for nodes 
            Arguments: 
                graph (graph object): graph with nodes to color
                color_by (str): attribute name to color nodes by
        """
        unique_label = list(np.unique(graph.vs[color_by]))

        if len(unique_label) > len(palette):
            raise ValueError(f"Dont have enough color, please provide more unique color: {len(unique_label)}")
        
        color_dict = dict(zip(unique_label, palette))
        nodes_color = [color_dict[attr] for attr in graph.vs[color_by]]

        return nodes_color
    
    def raw_graph(self, 
                color_nodes_by: str=None,
                label_nodes=None,
                palette=PALETTE):
        if color_nodes_by is not None:
            nodes_color = self._create_node_color(color_nodes_by, palette)
        else:
            nodes_color = None

        if label_nodes is not None:
            labels = self.partition.graph.vs[self.label_nodes]
        else:
            labels = None

        return ig.plot(self._igraph, 
                       vertex_label=labels, 
                       vertex_color=nodes_color)
    
    def _check_negative_value(self, weight_name: str) -> str:
        if f'abs_{weight_name}' in self._igraph.es.attribute_names():
            return f'abs_{weight_name}'
        
        if weight_name:
            if np.any(np.array(self._igraph.es[weight_name])<0):
                weight = f'abs_{weight_name}'
                self._igraph.es[weight] = list(map(abs, 
                                                    self._igraph.es[weight_name]))
            else:
                weight = weight_name
        return weight
    
    def _vertex_strength(self, weight_name:str=None):
        """ Calculate vertex strength 
            Arguments:
                weight_name: 
                    - name of edge attr to be used as weight (Default: None)
        """
        if weight_name:
            weight=self._check_negative_value(weight_name=weight_name)
        else:
            weight=weight_name

        self._igraph.vs['strength'] = self._igraph.strength(mode='all', loops=False, weights=weight)
    
    def _eigenvector_centrality(self, 
                               weight_name: str=None, 
                               scale: bool=True, 
                               col_name: str='evcent',
                               for_partition:bool=False,
                               partition_col: str=None) -> None:
        """ Calculate eigenvector_centrality for the graph
            Arguments: 
                - weight: name of edge attr want to use only support possitive weight. Will add edge attr as absolust of weight_name under column name 'abs_{weight_name}' (Default: None)
                - scale: scale the value to the maximun value [0-1]
                - name: name used to add tovertex df (Default: 'evcent')
                - for_partition: bool
                    True: calcucalate evcent for each partition/module
                    False: calcualate evcent score using the full graph
                - parition: str
                    name of the partition in vertex df. Implement if for_partition=True
                
        """
        
        if weight_name:
            weight= self._check_negative_value(weight_name)
        else: 
            weight = None
            
            
        if for_partition and not partition_col:
            raise ValueError("Please specify vertex attribute containing information of partition membership")
        
        #calcualte for each module in partition
        if for_partition and partition_col:
            partition = ig.VertexClustering(self._igraph, 
                                            membership=self._igraph.vs[partition_col])
            module_subgraph= partition.subgraphs()
            for graph in module_subgraph:
                score = graph.eigenvector_centrality(directed=self.directed,
                                                        scale=scale, 
                                                        weights=weight)
                #assign score into node/region    
                for index, node in enumerate(graph.vs):
                    self._igraph.vs.find(name=node['name'])[col_name] = score[index]

        else: #entire graph
            score = self._igraph.eigenvector_centrality(directed=self.directed,
                                                 scale=scale, 
                                                 weights=weight)
            self._igraph.vs[col_name] = score    
        

    def _betweenness_centrality(self, 
                               weight: str=None, 
                               scale: bool=True, 
                               name: str='betweenness',
                               for_partition:bool=False,
                               partition: str=None) -> None:
        
        if for_partition and not partition:
            raise ValueError("Please specify vertex attribute containing information of partition membership")
        
        if not for_partition:
            score = self._igraph.betweenness(directed=self.directed,
                                             weights=weight)
            self._igraph.vs[name] = score
        else:
            partition = ig.VertexClustering(self._igraph, 
                                            membership=self._igraph.vs[partition])
            module_subgraph= partition.subgraphs()
            for graph in module_subgraph:
                score = graph.betweenness(directed=self.directed,
                                        scale=scale, 
                                        weights=weight)
                #assign score into node/region    
                for index, node in enumerate(graph.vs):
                    self._igraph.vs.find(name=node['name'])[name] = score[index]

    
    def get_top_nodes(self,
                      rankCol: str,
                      n_top: int,
                      partition: str=None) -> pd.DataFrame:
        """ Calculate ranking score and Return name of regions with highest score.
            Arguments:
                rankCol (str):
                    name of score as vertex attr used to ran
                n_top (int): knumber of nodes with highest score to return
                partition:
                    name of vertex attr containing partition membership information -> will return top node in each partition
            Return:
                Return list of node names with n_top highest score

        """
        df = self.get_node_dataframe()

        if partition:
            nlargest_df = df.groupby(partition).apply(lambda x: x.nlargest(n_top, rankCol))
        else:
            nlargest_df = df.sort_values(rankCol, ascending=False).iloc[:n_top, :]
        return nlargest_df

    def _subgraph(self, 
                  rankCol: str,
                  n_top: int,
                  partition: str=None) -> ig.Graph:

        top_node_df = self.get_top_nodes(rankCol, n_top, partition)

        vertex_list = self._igraph.vs.select(name_in=top_node_df.name.tolist())

        subgraph = self._igraph.subgraph(vertex_list)
        return subgraph
        
            
    def partition_igraph(self, 
                        color_nodes_by: str, 
                        mark_groups: bool=True, 
                        layout: str='fruchterman_reingold',
                        label_nodes=None,
                        palette=PALETTE,
                        rankCol: str=None,
                        n_top: int=100,
                        partition: str=None,
                        **kwargs):
        """
            Visualize partition using igraph.plot()
            Default color nodes by module, mark_group=True

            Arguments:
                color_codes_by (str):
                    - Node attr to color by
                mark_group (bool):
                    - Default: True, hightlight different groups
                layout (str):
                    - Igraph supported layout
                label_nodes: 
                    - Choose node attributes to label in the graph
                    - Default: None
                rankCol (str):
                    name of score as vertex attr used to ran
                n_top (int): knumber of nodes with highest score to return
                partition:
                    name of vertex attr containing partition membership information -> will return top node in each partition
        """
        graph = self._subgraph(rankCol, n_top, partition)

        nodes_color = self._create_node_color(graph, color_nodes_by, palette)
        
        if label_nodes is not None:
            labels = graph.vs[self.label_nodes]
            print(labels)
        else:
            labels = None
        
        return ig.plot(graph, 
                        layout=layout,
                        mark_groups=mark_groups,                             mark_color=palette,
                        vertex_labels=labels,
                        vertex_color=nodes_color,
                        vertex_size=5,
                        hovermode='closest',
                        **kwargs)
    
    def partition_plotly(self, 
                        color_nodes_by: str='module',  
                        layout: str='fruchterman_reingold',
                        palette=PALETTE,
                        node_info=['name', 'module'],
                        fig_size= [1000, 1000],
                        rankType='evcent',
                        weight=None,
                        n_top=100,
                        fullGraph=False,
                        scale=False,
                        **kwargs):
        
        if self._igraph.vcount() > 1000:
            if self.partitioned:
                print("Number of vertice is too big. Plotting subgraph of top nodes for each partition")
                if rankType: 
                    pass
            

        graph = self._subgraph(rankType=rankType,
                               weight=weight, 
                               n_top=n_top,  
                               scale=scale, 
                               fullGraph=fullGraph)
        
        graph_layout = graph.graph.layout(layout)

        module_list = list(np.unique(graph.membership))

        nodes_color = self._create_node_color(graph.graph, color_nodes_by, palette)

        traces = []
        
        #node trace
        node_x = []
        node_y = []
        for node in range(len(graph.graph.vs)):
            x, y = graph_layout[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        hoverinfo='text',
                        marker=dict(size=12, 
                                    color=nodes_color,
                                    line=dict(
                                        color='black',
                                        width=2)),
                        )
        
        node_text = [] #node text when hover
        for node in graph.graph.vs:
            hovertext = ""
            for attr in node_info:
                hovertext += f"{attr}: {node[attr]} <br>"
            node_text.append(hovertext)
        node_trace.text = node_text

        #edge trace
        for edge in graph.graph.es:
            # Check if the edge connects nodes within the same community
            if graph.graph.vs[edge.source]['module'] == graph.graph.vs[edge.target]['module']:
                e_color = 'rgb(115,115,115)'  # Set a different color for edges within the same community
                e_width = 1  # Set a higher line width for edges within the same community
            else:
                e_color = 'rgb(189,189,189)'  # Default color for edges between different communities
                e_width = 0.5  # Default line width for edges between different communities

            x0, y0 = graph_layout[edge.source]
            x1, y1 = graph_layout[edge.target]
            traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(color=e_color, width=e_width),
                hoverinfo="none"
            ))
        
        #add node trace
        traces.append(node_trace)
        
        #figure
        fig = go.Figure(data=traces,
             layout=go.Layout(
                title='<br>Partition Network',
                titlefont_size=16,
                showlegend=False,)
                )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=fig_size[0], width=fig_size[1])
        return fig

    def get_node_dataframe(self):
        """ Return nodes/vertexes attributes as copy pandas dataframe """   
        df = self._igraph.get_vertex_dataframe().copy()
        return df
    
    def remove_vertex_attr(self, attr:str):
        """ Delete a vertex attr """
        del self._igraph.vs[attr]



        
        

        

        
        


        


        
