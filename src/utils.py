import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from pathlib import Path

# --- Set directories ---

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
PLOTS_DIR = ROOT_DIR / 'plots'


# --- WeightedStandardScaler Class ---

class WeightedStandardScaler:
    """
    Scales features using weighted mean and variance. Has same methods and attributes as sklearn's StandardScaler.

    Example
    -------
    >>> scaler = WeightedStandardScaler()
    >>> X_scaled = scaler.fit_transform(X, weights)
    >>> X_orig = scaler.inverse_transform(X_scaled)
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray, weights: np.ndarray):
        """Fits the scaler to the data X using sample weights."""
        # Allow weights to be (n_samples,) or (n_samples, n_features)
        if weights.ndim == 1:
            w = weights.reshape(-1, 1)
        else:
            w = weights
        self.mean_ = (X * w).sum(axis=0) / w.sum(axis=0)
        var = (w * (X - self.mean_)**2).sum(axis=0) / w.sum(axis=0)
        self.scale_ = np.sqrt(var)
        return self

    def transform(self, X: np.ndarray):
        """Transforms the data X using the fitted scaler."""
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, weights: np.ndarray):
        """Fits and transforms the data X using sample weights."""
        return self.fit(X, weights).transform(X)

    def inverse_transform(self, X: np.ndarray):
        """Undo the scaling of X: X_original = X_scaled * scale_ + mean_."""
        return X * self.scale_ + self.mean_

# --- WeightedPCA Class ---

class WeightedPCA:
    """
    Performs Principal Component Analysis using a weighted covariance matrix.
    Assumes input data `X_scaled` is already standardized using WeightedStandardScaler.
    Fits all components; transformation selects the top n.
    """
    def __init__(self):
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_features_ = None 

    def fit(self, X_scaled: np.ndarray, weights: np.ndarray):
        """
        Fits the Weighted PCA model to the standardized data X_scaled using sample weights. Computes and stores ALL principal components.
        """
        self.n_features_ = X_scaled.shape[1]

        # Calculate Weighted Covariance Matrix 
        sqrt_weights = np.sqrt(weights)
        weighted_X_scaled = X_scaled * sqrt_weights[:, np.newaxis]
        weighted_cov = (weighted_X_scaled.T @ weighted_X_scaled) / weights.sum() 

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Store all components and explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / total_variance
        self.components_ = eigenvectors.T # Store components as rows

        return self

    def transform(self, X_scaled: np.ndarray, n_components: int):
        """
        Applies the Weighted PCA transformation using the top n_components.
        Assumes X_scaled is already standardized using the SAME scaler used for fitting PCA.
        """
        # Select the top n components
        selected_components = self.components_[:n_components]

        # Project data onto the selected components
        X_transformed = X_scaled.dot(selected_components.T)

        return X_transformed

    def fit_transform(self, X_scaled: np.ndarray, weights: np.ndarray, n_components: int):
        """Fits PCA (all components) and transforms using the top n_components."""
        return self.fit(X_scaled, weights).transform(X_scaled, n_components)

    def get_explained_variance_ratio(self):
        """Returns the explained variance ratio for all components."""
        return self.explained_variance_ratio_
    

# --- Functions for computing and plotting correlation matrices ---

def weighted_corr_matrix(
        sub_df: pd.DataFrame,
        weights: np.ndarray,
        method: str = "spearman"
        ):
    """
    Compute weighted or unweighted correlation matrix for given method.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    weights (np.ndarray): Weights for each row.
    method (str): 'pearson' or 'spearman'.
    columns (list): List of columns to include.
    
    Returns:
    pd.DataFrame: The correlation matrix.
    """
    if method == 'spearman':
        sub_df = sub_df.rank()
    X = sub_df.values
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    
    # Compute weighted covariance matrix
    means = np.sum(X * weights[:, np.newaxis], axis=0)
    centered_X = X - means
    cov = (centered_X.T @ (centered_X * weights[:, np.newaxis]))

    # Normalize by outer product of standard deviations
    std_dev = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_dev, std_dev)

    # Return as DataFrame with original column names
    return pd.DataFrame(corr, index=sub_df.columns, columns=sub_df.columns)

def mean_mixed_corr_matrix(
        df: pd.DataFrame, 
        columns: List[str],
        weight_col: Optional[str] = None,
        method_lower: str = 'pearson', 
        method_upper: str = 'spearman', 
        years: List[int] = [2008, 2012, 2016, 2020]
        ):
    """
    For each year, computes correlation matrices using method_lower for lower triangle
    and method_upper for upper triangle, mixes them, and returns the average across years.
    
    Parameters:
    df (pd.DataFrame): The input dataframe with a 'year' column.
    weight_col (str, optional): Column to use as weights.
    method_lower (str): Correlation method for the lower triangle ('pearson' or 'spearman').
    method_upper (str): Correlation method for the upper triangle ('pearson' or 'spearman').
    years (List[int]): List of years to include in the average.
    
    Returns:
    pd.DataFrame: The mean mixed correlation matrix.
    """
    mixed_matrices = []
    
    for year in years:
        sub_df = df[df['year'] == year]
        sub_df = sub_df[columns]

        if weight_col is None:
            weights = np.ones(sub_df.shape[0])
        else:
            weights = sub_df[weight_col].values

        corr_lower = weighted_corr_matrix(sub_df, weights, method_lower)
        corr_upper = weighted_corr_matrix(sub_df, weights, method_upper)
        
        # Mix the matrices
        corr_mixed = corr_lower.copy()
        upper_mask = np.triu(np.ones_like(corr_mixed, dtype=bool), k=1)
        corr_mixed.values[upper_mask] = corr_upper.values[upper_mask]
        
        mixed_matrices.append(corr_mixed)
    
    mean_corr = np.mean([mat.values for mat in mixed_matrices], axis=0)
    return pd.DataFrame(mean_corr, index=columns, columns=columns)

def plot_dendrogram(
    distance_matrix,
    title="Feature Dendrogram",
    figsize=(15, 10),
    method='ward',
    save: bool = False
):
    """
    Plot a dendrogram for hierarchical clustering with orientation fixed to 'right'.
    Accepts only a distance matrix.
    If save=True, saves the plot to 'plots' folder in the root directory.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    import numpy as np

    condensed_distances = squareform(distance_matrix.values)
    linkage_matrix = linkage(condensed_distances, method=method)

    fig, ax = plt.subplots(figsize=figsize)
    color_threshold = 0.7 * linkage_matrix[:, 2].max()

    dendrogram(
        linkage_matrix,
        labels=distance_matrix.index.tolist(),
        ax=ax,
        orientation='right',
        leaf_rotation=0,
        leaf_font_size=9,
        color_threshold=color_threshold,
        above_threshold_color='gray'
    )

    for label in ax.get_yticklabels():
        label.set_horizontalalignment('right')
        label.set_verticalalignment('center')

    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_xlabel('Distance', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    max_dist = linkage_matrix[:, 2].max()
    ax.set_xticks(np.arange(0, max_dist + 0.5, 0.5), minor=True)
    ax.grid(True, alpha=0.3, axis='x', which='both')
    ax.set_xticks(np.arange(0, int(np.ceil(max_dist)) + 1, 1))
    ax.set_xticklabels([str(int(tick)) for tick in ax.get_xticks()])

    ax.axvline(x=color_threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Cluster Threshold ({color_threshold:.2f})')
    ax.legend()
    plt.tight_layout()
    if save:
        filename = PLOTS_DIR / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filename, bbox_inches='tight')

    plt.show()
    return fig, ax, linkage_matrix

def get_clusters_from_dendrogram(
        corr_matrix, 
        linkage_matrix, 
        criterion='distance', 
        t=0.7
        ):
    """
    Extract clusters from the dendrogram at a given threshold or number of clusters.

    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Original correlation matrix
    linkage_matrix : ndarray
        Linkage matrix from hierarchical clustering
    criterion : str, optional
        Criterion to use for forming clusters (passed to fcluster)
    t : float or int, optional
        Threshold or number of clusters (passed to fcluster)

    Returns:
    --------
    dict : Dictionary mapping cluster_id -> list of feature names
    """
    from scipy.cluster.hierarchy import fcluster

    cluster_labels = fcluster(
        Z=linkage_matrix, 
        t=t, 
        criterion=criterion
        )

    # Create cluster dictionary
    clusters = {}
    for i, feature in enumerate(corr_matrix.index):
        cluster_id = cluster_labels[i]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(feature)

    # Sort clusters by size (largest first)
    clusters = dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))

    return clusters

def print_cluster_summary(clusters, corr_matrix):
    """
    Print a summary of the clusters with their average internal correlations.
    
    Parameters:
    -----------
    clusters : dict
        Dictionary mapping cluster_id -> list of feature names
    corr_matrix : pd.DataFrame
        Original correlation matrix
    """
    print(f"Found {len(clusters)} clusters:\n")
    
    for cluster_id, features in clusters.items():
        print(f"Cluster {cluster_id} ({len(features)} features):")
        print(f"  Features: {', '.join(features)}")
        
        # Calculate average internal correlation based on absolute values
        cluster_corr = corr_matrix.loc[features, features].abs()
        # Get upper triangle (excluding diagonal)
        upper_tri = cluster_corr.where(
            np.triu(np.ones(cluster_corr.shape), k=1).astype(bool)
        )
        avg_abs_corr = upper_tri.stack().mean()
        min_abs_corr = upper_tri.stack().min()
        max_abs_corr = upper_tri.stack().max()
        print(f"  Min internal absolute correlation: {min_abs_corr:.3f}")
        print(f"  Average internal absolute correlation: {avg_abs_corr:.3f}")
        print(f"  Max internal absolute correlation: {max_abs_corr:.3f}")
        print()

def visualize_cluster_graph(
    clusters,
    corr_matrix,
    min_corr_threshold=0.0,
    *,
    unique_node_colors=False,
    show_node_labels=True,
    show_edge_labels=True,
    add_legend=False,
    cmap='tab20',
    save: bool = False
):
    """
    Visualize each cluster as a graph where nodes are features and edges are correlations.
    If save=True, saves the plot to 'plots' folder in the root directory.
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx is required for graph visualization. Install with: pip install networkx")
        return

    import matplotlib.patches as mpatches

    for cluster_id, features in clusters.items():
        if len(features) < 2:
            print(f"Cluster {cluster_id} has only {len(features)} feature(s), skipping graph.")
            continue

        # Create graph
        G = nx.Graph()
        G.add_nodes_from(features)

        # Add edges with correlations
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                corr = corr_matrix.loc[features[i], features[j]]
                if abs(corr) >= min_corr_threshold:
                    G.add_edge(features[i], features[j], weight=abs(corr), label=f"{corr:.2f}")

        if len(G.edges()) == 0:
            print(f"Cluster {cluster_id} has no edges above threshold {min_corr_threshold}, skipping.")
            continue

        # Layout
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Prepare node colors
        if unique_node_colors:
            # Use a colormap to generate enough distinct colors for all nodes
            cm = plt.get_cmap(cmap)
            colors = [cm(i / max(1, len(features) - 1)) for i in range(len(features))]
            node_color_map = {feat: colors[i] for i, feat in enumerate(features)}
            node_colors = [node_color_map[n] for n in G.nodes()]
        else:
            node_colors = 'lightblue'
            node_color_map = {feat: 'lightblue' for feat in features}

        plt.figure(figsize=(10, 10))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, alpha=0.9)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w * 5 for w in weights], alpha=0.6, edge_color='gray')
        if show_node_labels:
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        if show_edge_labels:
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        if add_legend:
            handles = [
                mpatches.Patch(color=node_color_map[f], label=f)
                for f in features
            ]
            plt.legend(
                handles=handles,
                title="Features",
                loc='center left',
                bbox_to_anchor=(1.02, 0.5),
                frameon=True
            )
        plt.title(
            f'Cluster {cluster_id} - Feature Correlation Graph\n({len(features)} features, {len(G.edges())} edges)',
            fontsize=14, fontweight='bold', pad=20
        )
        plt.axis('off')
        if add_legend:
            plt.tight_layout(rect=[0, 0, 0.82, 1])
        else:
            plt.tight_layout()
        if save:
            filename = PLOTS_DIR / f"cluster_{cluster_id}_graph.png"
            plt.savefig(filename, bbox_inches='tight')

        plt.show()


def create_euclidean_distance_matrix(corr_matrix: pd.DataFrame):
    """
    Accepts a symmetric correlation matrix, returns a distance matrix where each entry is the Euclidean distance 
    between the corresponding row and column vectors.
    """
    corr_values = corr_matrix.values
    n = corr_values.shape[0]
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i,n):
            d = np.sqrt(np.sum((corr_values[i, :] - corr_values[j, :])**2))
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d
            
    return pd.DataFrame(distance_matrix, index=corr_matrix.index, columns=corr_matrix.columns)

def plot_corr_heatmap(df_corr: pd.DataFrame, title: str, save: bool = False):
    """
    Plot the correlation heatmap from a given correlation matrix DataFrame.
    If save=True, saves the plot to 'plots' folder in the root directory.
    """
    cell_width = 0.5
    cell_height = 0.65
    box_width = cell_width * df_corr.shape[0]
    box_height = cell_height * df_corr.shape[1]

    fig = plt.figure(figsize=(box_width, box_height))
    ax = sns.heatmap(
        df_corr,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        square=False,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'}
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=20)

    ax.tick_params(
        labeltop=True, 
        labelbottom=True,
        labelright=True,
        labelleft=True
        )
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_horizontalalignment('center')
    for label in ax.get_yticklabels():
        label.set_rotation(0)
        label.set_verticalalignment('center')

    ax.figure.subplots_adjust(right=2.0)

    if save:
        filename = PLOTS_DIR / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filename, bbox_inches='tight')
        
    plt.show()