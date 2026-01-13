"""
Utility functions for visualization and data preprocessing
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_latent_space(features, labels, title, filename, figsize=(10, 8)):
    """
    Create t-SNE visualization of latent space
    
    Args:
        features: 2D array of features (already reduced to 2D)
        labels: Labels for coloring
        title: Plot title
        filename: Output filename
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    scatter = plt.scatter(features[:, 0], features[:, 1], 
                         c=labels, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_cluster_distribution(df, cluster_labels, title, filename):
    """
    Bar chart of language distribution per cluster
    
    Args:
        df: DataFrame with 'language' column
        cluster_labels: Cluster assignments
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(10, 6))
    
    cluster_lang = pd.DataFrame({
        'Cluster': cluster_labels,
        'Language': df['language'].values
    })
    
    counts = cluster_lang.groupby(['Cluster', 'Language']).size().unstack(fill_value=0)
    counts.plot(kind='bar', stacked=True)
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Language')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_training_loss(losses, title, filename, ylabel='Loss'):
    """
    Plot training loss curve
    
    Args:
        losses: List of loss values per epoch
        title: Plot title
        filename: Output filename
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_metrics_to_csv(results_list, filename):
    """
    Save metrics to CSV file
   
    Args:
        results_list: List of dictionaries with metrics
        filename: Output CSV filename
    """
    df = pd.DataFrame(results_list)
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")
