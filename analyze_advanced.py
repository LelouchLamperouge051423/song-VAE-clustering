"""
Advanced Analysis Script
- Multi-modal clustering (Audio + Text + Artist)
- Metrics: Silhouette, NMI, ARI, Cluster Purity
- Visualizations: Latent space, cluster distribution, reconstructions
- Baseline comparisons: VAE, CVAE, Beta-VAE, AE, PCA, Spectral
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, 
    normalized_mutual_info_score, 
    adjusted_rand_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from train_vae import VAE, LATENT_DIM, INPUT_DIM, SEQ_LEN, DEVICE
from train_cvae import CVAE, SimpleAutoencoder, SongDatasetWithLabels

# Config
DATASET_INDEX = PROJECT_ROOT / "outputs" / "dataset_index.csv"
K_CLUSTERS = 2
TEXT_DIM = 32
ARTIST_DIM = 20

# =========================
# METRICS
# =========================
def cluster_purity(labels_true, labels_pred):
    """Compute cluster purity score"""
    contingency = {}
    for true, pred in zip(labels_true, labels_pred):
        if pred not in contingency:
            contingency[pred] = Counter()
        contingency[pred][true] += 1
    
    total_correct = sum(max(counter.values()) for counter in contingency.values())
    return total_correct / len(labels_true)

def compute_all_metrics(labels_true, labels_pred, features):
    """Compute all clustering metrics"""
    # Handle edge case of single cluster
    unique_clusters = len(set(labels_pred))
    if unique_clusters < 2:
        return {
            'Silhouette': -1,
            'NMI': 0,
            'ARI': 0,
            'Purity': 0,
            'n_clusters': unique_clusters
        }
    
    return {
        'Silhouette': silhouette_score(features, labels_pred),
        'NMI': normalized_mutual_info_score(labels_true, labels_pred),
        'ARI': adjusted_rand_score(labels_true, labels_pred),
        'Purity': cluster_purity(labels_true, labels_pred),
        'n_clusters': unique_clusters
    }

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features():
    """Extract all feature types from dataset"""
    print("Loading dataset...")
    df = pd.read_csv(DATASET_INDEX)
    
    # --- Text Features (TF-IDF on Title+Artist) ---
    print("Extracting text features...")
    df['text_content'] = df['title'].fillna('') + " " + df['artist'].fillna('')
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    text_raw = tfidf.fit_transform(df['text_content']).toarray()
    pca_text = PCA(n_components=TEXT_DIM, random_state=42)
    text_features = pca_text.fit_transform(text_raw)
    
    # --- Artist Features (Encoded) ---
    print("Extracting artist features...")
    le = LabelEncoder()
    artist_encoded = le.fit_transform(df['artist'].fillna('Unknown'))
    # One-hot encode top artists, rest as "Other"
    top_n = ARTIST_DIM
    artist_counts = Counter(artist_encoded)
    top_artists = [a for a, _ in artist_counts.most_common(top_n)]
    artist_features = np.zeros((len(df), top_n))
    for i, a in enumerate(artist_encoded):
        if a in top_artists:
            artist_features[i, top_artists.index(a)] = 1.0
    
    # --- Audio Features (Multiple Models) ---
    audio_features = {}
    valid_indices = []
    
    # Load models
    models = {}
    
    # VAE
    if (PROJECT_ROOT / "vae_model.pth").exists():
        vae = VAE(input_dim=INPUT_DIM, hidden_dim=LATENT_DIM).to(DEVICE)
        vae.load_state_dict(torch.load(PROJECT_ROOT / "vae_model.pth", map_location=DEVICE))
        vae.eval()
        models['VAE'] = vae
    
    # CVAE
    if (PROJECT_ROOT / "cvae_model.pth").exists():
        cvae = CVAE().to(DEVICE)
        cvae.load_state_dict(torch.load(PROJECT_ROOT / "cvae_model.pth", map_location=DEVICE))
        cvae.eval()
        models['CVAE'] = cvae
    
    # Beta-VAE
    if (PROJECT_ROOT / "beta_vae_model.pth").exists():
        beta_vae = CVAE().to(DEVICE)
        beta_vae.load_state_dict(torch.load(PROJECT_ROOT / "beta_vae_model.pth", map_location=DEVICE))
        beta_vae.eval()
        models['Beta-VAE'] = beta_vae
    
    # Autoencoder
    if (PROJECT_ROOT / "autoencoder_model.pth").exists():
        ae = SimpleAutoencoder().to(DEVICE)
        ae.load_state_dict(torch.load(PROJECT_ROOT / "autoencoder_model.pth", map_location=DEVICE))
        ae.eval()
        models['Autoencoder'] = ae
    
    # Initialize storage
    for name in models:
        audio_features[name] = []
    audio_features['Raw'] = []  # For spectral baseline
    
    language_map = {'english': 0, 'bangla': 1}
    
    print("Extracting latent features from all models...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        feat_path = Path(row['features'])
        if not feat_path.exists():
            continue
        
        try:
            mfcc = np.load(feat_path)
            if mfcc.ndim != 2 or mfcc.shape[0] != INPUT_DIM:
                continue
            
            # Preprocess
            if mfcc.shape[1] < SEQ_LEN:
                mfcc = np.pad(mfcc, ((0, 0), (0, SEQ_LEN - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :SEQ_LEN]
            
            mean = mfcc.mean(axis=1, keepdims=True)
            std = mfcc.std(axis=1, keepdims=True) + 1e-8
            mfcc_norm = (mfcc - mean) / std
            
            data_tensor = torch.FloatTensor(mfcc_norm).unsqueeze(0).to(DEVICE)
            lang = row['language'].strip().lower()
            label = language_map.get(lang, 0)
            label_tensor = torch.LongTensor([label]).to(DEVICE)
            
            with torch.no_grad():
                # VAE
                if 'VAE' in models:
                    _, mu, _ = models['VAE'](data_tensor)
                    audio_features['VAE'].append(mu.cpu().numpy().flatten())
                
                # CVAE
                if 'CVAE' in models:
                    _, mu, _ = models['CVAE'](data_tensor, label_tensor)
                    audio_features['CVAE'].append(mu.cpu().numpy().flatten())
                
                # Beta-VAE
                if 'Beta-VAE' in models:
                    _, mu, _ = models['Beta-VAE'](data_tensor, label_tensor)
                    audio_features['Beta-VAE'].append(mu.cpu().numpy().flatten())
                
                # Autoencoder
                if 'Autoencoder' in models:
                    _, z = models['Autoencoder'](data_tensor)
                    audio_features['Autoencoder'].append(z.cpu().numpy().flatten())
            
            # Raw features for spectral baseline
            audio_features['Raw'].append(mfcc_norm.flatten()[:1000])  # Truncate for memory
            valid_indices.append(idx)
            
        except Exception as e:
            continue
    
    # Convert to arrays
    for name in audio_features:
        if audio_features[name]:
            audio_features[name] = np.array(audio_features[name])
        else:
            audio_features[name] = None
    
    # Filter other features
    text_features = text_features[valid_indices]
    artist_features = artist_features[valid_indices]
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    
    return df_valid, audio_features, text_features, artist_features

# =========================
# MULTI-MODAL FUSION
# =========================
def create_multimodal_features(audio_feat, text_feat, artist_feat):
    """Combine all modalities"""
    scaler = StandardScaler()
    
    audio_scaled = scaler.fit_transform(audio_feat)
    text_scaled = scaler.fit_transform(text_feat)
    artist_scaled = scaler.fit_transform(artist_feat)
    
    return np.hstack([audio_scaled, text_scaled, artist_scaled])

# =========================
# VISUALIZATIONS
# =========================
def plot_latent_space(features, labels, title, filename):
    """t-SNE visualization of latent space"""
    print(f"Creating t-SNE plot: {title}")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedding = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                         c=labels, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_cluster_distribution(df, cluster_labels, title, filename):
    """Bar chart of language distribution per cluster"""
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

def plot_reconstruction_examples(model, dataset, num_examples=4, filename='reconstruction_examples.png'):
    """Show original vs reconstructed MFCCs"""
    model.eval()
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 3*num_examples))
    
    for i in range(num_examples):
        data, label = dataset[i]
        data = data.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            if hasattr(model, 'decode'):  # CVAE
                label_tensor = torch.LongTensor([label]).to(DEVICE)
                recon, _, _ = model(data, label_tensor)
            else:  # VAE
                recon, _, _ = model(data)
        
        orig = data.cpu().numpy()[0]
        rec = recon.cpu().numpy()[0]
        
        axes[i, 0].imshow(orig, aspect='auto', origin='lower')
        axes[i, 0].set_title(f'Original (Sample {i+1})')
        axes[i, 1].imshow(rec, aspect='auto', origin='lower')
        axes[i, 1].set_title(f'Reconstructed (Sample {i+1})')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# =========================
# MAIN ANALYSIS
# =========================
def main():
    print("="*60)
    print("ADVANCED VAE ANALYSIS")
    print("="*60)
    
    # Extract features
    df, audio_features, text_features, artist_features = extract_features()
    labels_true = LabelEncoder().fit_transform(df['language'])
    
    # Results storage
    all_results = []
    
    # --- Baseline: PCA + K-Means ---
    print("\n--- PCA + K-Means Baseline ---")
    if audio_features['Raw'] is not None:
        pca = PCA(n_components=LATENT_DIM, random_state=42)
        pca_features = pca.fit_transform(audio_features['Raw'])
        labels_pca = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10).fit_predict(pca_features)
        metrics_pca = compute_all_metrics(labels_true, labels_pca, pca_features)
        metrics_pca['Method'] = 'PCA + K-Means'
        all_results.append(metrics_pca)
        plot_latent_space(pca_features, labels_true, "PCA Latent Space", "pca_latent.png")
    
    # --- Baseline: Direct Spectral (Raw K-Means) ---
    print("\n--- Direct Spectral Clustering ---")
    if audio_features['Raw'] is not None:
        labels_raw = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10).fit_predict(audio_features['Raw'])
        metrics_raw = compute_all_metrics(labels_true, labels_raw, audio_features['Raw'])
        metrics_raw['Method'] = 'Spectral + K-Means'
        all_results.append(metrics_raw)
    
    # --- VAE Models ---
    for model_name in ['VAE', 'Autoencoder', 'CVAE', 'Beta-VAE']:
        if audio_features.get(model_name) is not None:
            print(f"\n--- {model_name} ---")
            feats = audio_features[model_name]
            labels_pred = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10).fit_predict(feats)
            metrics = compute_all_metrics(labels_true, labels_pred, feats)
            metrics['Method'] = f'{model_name} + K-Means'
            all_results.append(metrics)
            plot_latent_space(feats, labels_true, f"{model_name} Latent Space (Ground Truth)", f"{model_name.lower()}_latent_gt.png")
            plot_latent_space(feats, labels_pred, f"{model_name} Latent Space (Clusters)", f"{model_name.lower()}_latent_cluster.png")
            plot_cluster_distribution(df, labels_pred, f"{model_name} Cluster Distribution", f"{model_name.lower()}_cluster_dist.png")
    
    # --- Multi-Modal (Best Audio + Text + Artist) ---
    print("\n--- Multi-Modal Clustering ---")
    best_audio = None
    for name in ['CVAE', 'Beta-VAE', 'VAE', 'Autoencoder']:
        if audio_features.get(name) is not None:
            best_audio = audio_features[name]
            best_name = name
            break
    
    if best_audio is not None:
        multimodal = create_multimodal_features(best_audio, text_features, artist_features)
        labels_mm = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10).fit_predict(multimodal)
        metrics_mm = compute_all_metrics(labels_true, labels_mm, multimodal)
        metrics_mm['Method'] = f'Multi-Modal ({best_name}+Text+Artist)'
        all_results.append(metrics_mm)
        plot_latent_space(multimodal, labels_true, "Multi-Modal Features", "multimodal_latent.png")
        plot_cluster_distribution(df, labels_mm, "Multi-Modal Cluster Distribution", "multimodal_cluster_dist.png")
    
    # --- Print Results Table ---
    print("\n" + "="*90)
    print(f"{'Method':<35} | {'Silhouette':>10} | {'NMI':>8} | {'ARI':>8} | {'Purity':>8}")
    print("-"*90)
    for r in all_results:
        print(f"{r['Method']:<35} | {r['Silhouette']:>10.4f} | {r['NMI']:>8.4f} | {r['ARI']:>8.4f} | {r['Purity']:>8.4f}")
    print("="*90)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("advanced_metrics.csv", index=False)
    
    with open("advanced_metrics.txt", "w") as f:
        f.write(f"{'Method':<35} | {'Silhouette':>10} | {'NMI':>8} | {'ARI':>8} | {'Purity':>8}\n")
        f.write("-"*90 + "\n")
        for r in all_results:
            f.write(f"{r['Method']:<35} | {r['Silhouette']:>10.4f} | {r['NMI']:>8.4f} | {r['ARI']:>8.4f} | {r['Purity']:>8.4f}\n")
    
    # --- Reconstruction Examples ---
    print("\nGenerating reconstruction examples...")
    dataset = SongDatasetWithLabels(DATASET_INDEX)
    
    if (PROJECT_ROOT / "cvae_model.pth").exists():
        cvae = CVAE().to(DEVICE)
        cvae.load_state_dict(torch.load(PROJECT_ROOT / "cvae_model.pth", map_location=DEVICE))
        plot_reconstruction_examples(cvae, dataset, filename='cvae_reconstructions.png')
    elif (PROJECT_ROOT / "vae_model.pth").exists():
        vae = VAE(input_dim=INPUT_DIM, hidden_dim=LATENT_DIM).to(DEVICE)
        vae.load_state_dict(torch.load(PROJECT_ROOT / "vae_model.pth", map_location=DEVICE))
        plot_reconstruction_examples(vae, dataset, filename='vae_reconstructions.png')
    
    print("\nAnalysis complete! Results saved to advanced_metrics.csv and .txt")

if __name__ == "__main__":
    main()
