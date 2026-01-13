"""
Evaluation metrics for clustering quality
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from collections import Counter


def cluster_purity(labels_true, labels_pred):
    """
    Compute cluster purity score
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
    
    Returns:
        float: Purity score between 0 and 1
    """
    contingency = {}
    for true, pred in zip(labels_true, labels_pred):
        if pred not in contingency:
            contingency[pred] = Counter()
        contingency[pred][true] += 1
    
    total_correct = sum(max(counter.values()) for counter in contingency.values())
    return total_correct / len(labels_true)


def compute_all_metrics(labels_true, labels_pred, features):
    """
    Compute all clustering quality metrics
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels  
        features: Feature matrix used for clustering
    
    Returns:
        dict: Dictionary containing all metrics
    """
    unique_clusters = len(set(labels_pred))
    
    # Handle edge case
    if unique_clusters < 2:
        return {
            'Silhouette': -1,
            'NMI': 0,
            'ARI': 0,
            'Purity': 0,
            'Calinski-Harabasz': 0,
            'Davies-Bouldin': -1,
            'n_clusters': unique_clusters
        }
    
    return {
        'Silhouette': silhouette_score(features, labels_pred),
        'NMI': normalized_mutual_info_score(labels_true, labels_pred),
        'ARI': adjusted_rand_score(labels_true, labels_pred),
        'Purity': cluster_purity(labels_true, labels_pred),
        'Calinski-Harabasz': calinski_harabasz_score(features, labels_pred),
        'Davies-Bouldin': davies_bouldin_score(features, labels_pred),
        'n_clusters': unique_clusters
    }


def print_metrics(metrics_dict, method_name=""):
    """Pretty print metrics"""
    if method_name:
        print(f"\n{method_name}")
        print("=" * 50)
    
    for metric, value in metrics_dict.items():
        if metric != 'n_clusters':
            print(f"{metric:.<30} {value:>10.4f}")
    print(f"{'Number of clusters':.<30} {metrics_dict['n_clusters']:>10}")


def compare_methods(results_list):
    """
    Compare multiple clustering methods
    
    Args:
        results_list: List of dicts with 'Method' and metric keys
    
    Returns:
        None (prints comparison table)
    """
    print("\n" + "=" * 90)
    print(f"{'Method':<35} | {'Silhouette':>10} | {'NMI':>8} | {'ARI':>8} | {'Purity':>8}")
    print("-" * 90)
    
    for result in results_list:
        method = result.get('Method', 'Unknown')
        sil = result.get('Silhouette', 0)
        nmi = result.get('NMI', 0)
        ari = result.get('ARI', 0)
        purity = result.get('Purity', 0)
        
        print(f"{method:<35} | {sil:>10.4f} | {nmi:>8.4f} | {ari:>8.4f} | {purity:>8.4f}")
    
    print("=" * 90)
