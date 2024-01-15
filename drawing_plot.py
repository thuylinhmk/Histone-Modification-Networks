import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from matplotlib.pyplot import gcf
import matplotlib.colors as mcolors
import umap

PALETTE=sns.color_palette('tab20').as_hex()

def umap2D_drawing_cluster(data, 
                           n_components:int=2,
                           n_neighbors: int =10,
                           random_state=218,
                           draw_components=[0,1],
                           cluster_info:bool=True):
    if len(draw_components) != 2:
        raise ValueError("Input a list which components you want to plot. Eg: [0,1] to draw C1 and C2")
    df = data.copy()

    if cluster_info:
        # Get the cluster assignments for each data point
        cluster_assignments = df.pop('Cluster').to_list()
        unique_labels = np.unique(cluster_assignments)

    # Perform dimensionality reduction with UMAP
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
    embeddings = reducer.fit_transform(df.values)


    # Create a color scale for the clusters
    color_scale = [mcolors.to_hex(c) for c in mcolors.BASE_COLORS.values()]

    # Map the labels to the corresponding color
    #color_labels = [color_scale[label] for label in cluster_assignments]
    
    traces = []
    for label in unique_labels:
        # Filter the data points belonging to the current label
        cluster_points = embeddings[cluster_assignments == label]

        # Create a scatter plot trace for the current label
        trace = go.Scatter(
            x=cluster_points[:, draw_components[0]],  # UMAP component 1
            y=cluster_points[:, draw_components[1]],  # UMAP component 2 
            mode='markers',
            name=f'Cluster {label}',
            marker=dict(
                size=5,
                color=color_scale[label],
                opacity=0.8
            )
        )
        traces.append(trace)

    # Create the layout with legend
    layout = go.Layout(
        xaxis_title=f'UMAP Component {draw_components[0]+1}',
        yaxis_title=f'UMAP Component {draw_components[1]+1}',
        title='UMAP 2D Visualization',
        showlegend=True
    )

    # Create the figure with traces and layout
    fig = go.Figure(data=traces, layout=layout)
    # Display the interactive plot
    return fig
    

def umap3D_drawing_cluster(data,
                   n_components:int=3,
                   n_neighbors: int =10, 
                   random_state=218,
                   draw_components=[0,1,2], 
                   cluster_info:bool=True):
    if len(draw_components) !=3:
        raise ValueError("draw_components is a list of 3 int for wanted PC")
    df = data.copy()

    if cluster_info:
        # Get the cluster assignments for each data point
        cluster_assignments = df.pop('Cluster').to_list()
        unique_labels = np.unique(cluster_assignments)

    # Perform dimensionality reduction with UMAP
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
    embeddings = reducer.fit_transform(df.values)


    # Create a color scale for the clusters
    color_scale = [mcolors.to_hex(c) for c in mcolors.BASE_COLORS.values()]

    # Map the labels to the corresponding color
    #color_labels = [color_scale[label] for label in cluster_assignments]
    
    traces = []
    for label in unique_labels:
        # Filter the data points belonging to the current label
        cluster_points = embeddings[cluster_assignments == label]

        # Create a scatter plot trace for the current label
        trace = go.Scatter3d(
            x=cluster_points[:, draw_components[0]],  # UMAP component 1
            y=cluster_points[:, draw_components[1]],  # UMAP component 2
            z=cluster_points[:, draw_components[2]],  # UMAP component 3
            mode='markers',
            name=f'Cluster {label}',
            marker=dict(
                size=5,
                color=color_scale[label],
                opacity=0.8
            )
        )
        traces.append(trace)

    # Create the layout with legend
    layout = go.Layout(
        scene=dict(
            xaxis_title=f'UMAP Component {draw_components[0]+1}',
            yaxis_title=f'UMAP Component {draw_components[1]+1}',
            zaxis_title=f'UMAP Component {draw_components[2]+1}'
        ),
        title='UMAP 3D Visualization',
        showlegend=True
    )

    # Create the figure with traces and layout
    fig = go.Figure(data=traces, layout=layout)
    # Display the interactive plot
    return fig


def draw_clustermap(data, 
                    row_label=None, 
                    col_label=None, 
                    row_label_color=None, 
                    col_label_color=None, 
                    col_cluster=False, 
                    row_cluster=True):
    if row_label_color is None and row_label is not None: 
        row_label_color = mcolors.BASE_COLORS
    if col_label_color is None and col_label is not None:
        col_label_color = sns.color_palette("icefire", len(col_label.unique()))

    if row_label is not None:
        # Create a color palette for the row labels
        row_palette = dict(zip(row_label.unique(), row_label_color))
        row_colors = row_label.map(row_palette)

    if col_label is not None:
        col_palette = dict(zip(col_label.unique(), col_label_color))
        col_colors = col_label.map(col_palette)

    

    # Generate the clustermap
    cluster_map = sns.clustermap(data, 
                                col_cluster=col_cluster, 
                                row_cluster=row_cluster,
                                cmap="magma",
                                method="ward",
                                row_colors=row_colors,
                                col_colors=col_colors,
                                )
    if row_label is not None: 
        for label in row_label.unique():
            cluster_map.ax_col_dendrogram.bar(0, 0, color=row_palette[label], label=label, linewidth=0);
        l1 = cluster_map.ax_col_dendrogram.legend(title='Cluster', 
                                            loc="center",
                                            ncol=2,
                                            bbox_to_anchor=(0.25, 0.89),
                                            bbox_transform=gcf().transFigure)
    if col_label is not None:
        for label in col_label.unique():
            cluster_map.ax_row_dendrogram.bar(0, 0, color=col_palette[label], label=label, linewidth=0);
        l2 = cluster_map.ax_row_dendrogram.legend(title='Tissue', loc="center", ncol=2, bbox_to_anchor=(0.66, 0.89), bbox_transform=gcf().transFigure)

    # Show the plot
    plt.show()

class GraphDF():
    """ Graph related analysis. Takes in graph data representation of the VNC network,
        or subsets of it. For now, supports only iGraph due to large network.
        Coupled to conndf for now, need to uncouple by making an aesthetics
        class. """
    def __init__(self, graph, weight_name, label_name, profile):
        self.graph = graph
        # iGraph and community vars
        self.profile = profile
        self.vertex_df = self.graph.get_vertex_dataframe()
        self.vertex_df['comm_attr'] = self.profile.membership #annotate nodes by assigned community
        self.unique_communities = self.vertex_df['comm_attr'].unique()
        self.node_list = None
        self.weight_name = weight_name
        self.label_name = label_name

    def annotate_nodes(self, ctype='strength'):
        """ Annotates nodes by graph metrics such as strength (weighted degree)
        or betweenness centrality scores. Used for ranking nodes, and sampling 
        the top N nodes in method sample. Can annotate a given vertex_df by 
        multiple centrality types. 

        Args (currently supported): # should change this to pass methods
            'strength'
            'betweenness'
        """

        _df = None
        for comm in self.unique_communities:
            comm_nodes = list(self.vertex_df.query(f'comm_attr == {comm}').index)
            comm_subgraph = self.graph.subgraph(comm_nodes)

            if ctype == 'strength':
                wc_strength = comm_subgraph.strength(weights=self.weight_name)
                add_dict = {'wc_strength': wc_strength}

            elif ctype == 'betweenness':
                wc_betweenness = comm_subgraph.betweenness(weights=self.weight_name)
                add_dict = {'wc_betweenness': wc_betweenness}

            temp_df = pd.DataFrame(add_dict, index=comm_nodes)
            if _df is None:
                _df = temp_df
            else:
                _df = pd.concat([_df, temp_df])
        self.vertex_df = self.vertex_df.join(_df)
    
    def sample(self, n, ctype='strength'):
        """ Gets the top n nodes ranked by centrality type. Invokes node
        annotation method, annotate_nodes if not centrality type does not exist
        in the current vertex_df. Used for constructing a subgraph. """
        self.node_list = []

        if f'wc_{ctype}' not in self.vertex_df.columns:
            self.annotate_nodes(ctype=ctype)

        for comm in self.unique_communities:
            comm_nodes = list(
                self.vertex_df.query(
                    f'comm_attr == {comm}'
                    ).sort_values(by=f'wc_{ctype}', ascending=False).head(n).index
                )
            self.node_list += comm_nodes

    def get_manual_partition(self, n, ctype='strength'):
        # check if annotated by ctype
        if f'wc_{ctype}' not in self.vertex_df.columns:
            self.annotate_nodes(ctype=ctype)
        self.sample(n, ctype)

        manual_partition = ig.VertexClustering(
            self.graph.subgraph(sorted(self.node_list)),
            list(self.vertex_df.loc[sorted(self.node_list)]['comm_attr']))
        return manual_partition

    def plot_by_class(
        self,
        n=30,
        manual_partition=None,
        label_nodes=False,
        ctype='strength',
        layout='fruchterman_reingold',
        vertex_size=5,
        edge_width=0.1,
        edge_arrow_size=0.1,
        edge_size=0.1,
        **kwargs):
        """ iGraph plot of subgraph network with matplotlib backend. **kwargs
        passed to ig.plot function. MDS layout by default. Has reasonable 
        ig.plot default vertex and edge parameters. Calls annotate_nodes and
        sample class methods, combining functionality to produce plot.
        """
        
        # Manual partition with VertexClustering object for shading aesthetic
        if manual_partition is None:
            manual_partition = self.get_manual_partition(n, ctype)

        if label_nodes:
            labels = manual_partition.graph.vs[self.label_name]
        else:
            labels = None

        #edge_colors = ['orange' if weight > 0 else 'grey' for weight in manual_partition.vcount]

        # Plot
        return ig.plot(
            manual_partition,
            mark_groups=True, # Shade by community
            layout=layout,
            vertex_size=vertex_size,
            edge_width=edge_width,
            edge_arrow_size=edge_arrow_size,
            edge_size=edge_size,
            vertex_label=labels,
            hovermode='closest',
            **kwargs
            )
