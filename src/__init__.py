"""
Song Dataset Project - VAE-based Music Clustering
"""

__version__ = "1.0.0"

from .vae import VAE, CVAE, SimpleAutoencoder, vae_loss, ae_loss
from .dataset import SongDataset, SongDatasetWithLabels
from .clustering import (
    extract_latent_features,
    apply_kmeans,
    apply_agglomerative,
    apply_dbscan,
    reduce_tsne,
    reduce_pca,
    extract_text_features,
    fuse_multimodal
)
from .evaluation import (
    cluster_purity,
    compute_all_metrics,
    print_metrics,
    compare_methods
)
from .utils import (
    plot_latent_space,
    plot_cluster_distribution,
    plot_training_loss,
    save_metrics_to_csv
)

__all__ = [
    'VAE', 'CVAE', 'SimpleAutoencoder', 'vae_loss', 'ae_loss',
    'SongDataset', 'SongDatasetWithLabels',
    'extract_latent_features', 'apply_kmeans', 'apply_agglomerative', 'apply_dbscan',
    'reduce_tsne', 'reduce_pca', 'extract_text_features', 'fuse_multimodal',
    'cluster_purity', 'compute_all_metrics', 'print_metrics', 'compare_methods',
    'plot_latent_space', 'plot_cluster_distribution', 'plot_training_loss', 'save_metrics_to_csv'
]
