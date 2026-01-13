"""
Clustering algorithms and feature extraction
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch


def extract_latent_features(model, dataloader, device):
    """Extract latent features from VAE model"""
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            
            # Handle different model types
            if hasattr(model, 'encode'):  # CVAE
                label_tensor = torch.LongTensor([label] * len(data)).to(device)
                mu, _ = model.encode(data, label_tensor)
            else:  # VAE or Autoencoder
                if hasattr(model, 'reparameterize'):  # VAE
                    _, mu, _ = model(data)
                else:  # Autoencoder
                    _, mu = model(data)
            
            latents.append(mu.cpu().numpy())
            labels.extend(label if isinstance(label, list) else [label])
   
    return np.concatenate(latents, axis=0), np.array(labels)


def apply_kmeans(features, n_clusters=2, random_state=42):
    """Apply K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels


def apply_agglomerative(features, n_clusters=2):
    """Apply Agglomerative Clustering"""
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(features)
    return labels


def apply_dbscan(features, eps=5.0, min_samples=5):
    """Apply DBSCAN clustering"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)
    return labels


def reduce_tsne(features, n_components=2, random_state=42):
    """Reduce dimensions using t-SNE"""
    tsne = TSNE(n_components=n_components, random_state=random_state, 
                init='pca', learning_rate='auto')
    embedding = tsne.fit_transform(features)
    return embedding


def reduce_pca(features, n_components=64, random_state=42):
    """Reduce dimensions using PCA"""
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(features)
    return reduced


def extract_text_features(titles, artists, n_components=32):
    """Extract TF-IDF features from text"""
    # Combine title and artist
    text_content = [f"{t} {a}" for t, a in zip(titles, artists)]
    
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    text_embeddings = tfidf.fit_transform(text_content).toarray()
    
    # PCA reduction
    pca = PCA(n_components=n_components, random_state=42)
    text_features = pca.fit_transform(text_embeddings)
    
    # Normalize
    scaler = StandardScaler()
    text_features = scaler.fit_transform(text_features)
    
    return text_features


def fuse_multimodal(audio_features, text_features, artist_features=None):
    """Fuse multiple modalities"""
    scaler = StandardScaler()
    
    audio_scaled = scaler.fit_transform(audio_features)
    text_scaled = scaler.fit_transform(text_features)
    
    if artist_features is not None:
        artist_scaled = scaler.fit_transform(artist_features)
        return np.hstack([audio_scaled, text_scaled, artist_scaled])
    else:
        return np.hstack([audio_scaled, text_scaled])
