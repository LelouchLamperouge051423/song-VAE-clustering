
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Add project root to path to import train_vae
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Import VAE model definition
from train_vae import VAE, LATENT_DIM, INPUT_DIM, SEQ_LEN, DEVICE

# Config
DATASET_INDEX = PROJECT_ROOT / "outputs" / "dataset_index.csv"
MODEL_PATH = PROJECT_ROOT / "vae_model.pth"
TEXT_DIM = 32 # Dimension for text features after PCA
HYBRID_DIM = LATENT_DIM + TEXT_DIM

def load_data_and_features():
    print("Loading dataset index...")
    df = pd.read_csv(DATASET_INDEX)
    
    # --- Text Features ---
    print("Generating Text Features (TF-IDF + PCA)...")
    # Combine title and artist
    df['text_content'] = df['title'].fillna('') + " " + df['artist'].fillna('')
    
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    text_embeddings = tfidf.fit_transform(df['text_content']).toarray()
    
    # PCA for Text
    pca = PCA(n_components=TEXT_DIM, random_state=42)
    text_reduced = pca.fit_transform(text_embeddings)
    
    # Normalize Text Features
    scaler_text = StandardScaler()
    text_features = scaler_text.fit_transform(text_reduced)
    
    # --- Audio Features (VAE Latent) ---
    print("Loading VAE Model...")
    # Matches train_vae.py: def __init__(self, input_dim=INPUT_DIM, hidden_dim=LATENT_DIM):
    model = VAE(input_dim=INPUT_DIM, hidden_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    audio_features = []
    valid_indices = []
    
    print("Extracting Audio Latent Features...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        feat_path = Path(row['features'])
        if not feat_path.exists():
            continue
            
        try:
            mfcc = np.load(feat_path) # Shape: (n_mfcc, T)
            
            # Validate input shape
            if mfcc.ndim != 2 or mfcc.shape[0] != INPUT_DIM:
                continue
            
            # Preprocessing (same as training)
            if mfcc.shape[1] < SEQ_LEN:
                pad_width = SEQ_LEN - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :SEQ_LEN]
            
            # Normalize
            mean = mfcc.mean(axis=1, keepdims=True)
            std = mfcc.std(axis=1, keepdims=True) + 1e-8
            mfcc = (mfcc - mean) / std
            
            # To Tensor
            data_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(DEVICE) # (1, 40, 430)
            
            with torch.no_grad():
                # VAE forward returns (recon, mu, logvar)
                _, mu, _ = model(data_tensor)
                latent = mu.cpu().numpy().flatten()
                
                # Validate latent dimension
                if latent.shape[0] == LATENT_DIM:
                    audio_features.append(latent)
                    valid_indices.append(idx)
                
        except Exception as e:
            print(f"Error processing {feat_path}: {e}")
            
    audio_features = np.array(audio_features)
    
    # Filter text features to match valid audio indices
    text_features = text_features[valid_indices]
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    
    # --- Hybrid Features ---
    print("Creating Hybrid Features...")
    # Concatenate Audio (64) + Text (32)
    hybrid_features = np.hstack([audio_features, text_features])
    
    return df_valid, audio_features, hybrid_features

def run_clustering(features, labels_true, method_name, k=2):
    results = {}
    
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(features)
    results['KMeans'] = evaluate_clustering(features, labels_kmeans, labels_true)
    
    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=k)
    labels_agg = agg.fit_predict(features)
    results['Agglomerative'] = evaluate_clustering(features, labels_agg, labels_true)
    
    # DBSCAN
    # Eps is tricky, trying a heuristic or fixed value. 
    # For normalized high-dim data, 0.5-5.0 range often used. Let's try 5.0 for 96-dim
    dbscan = DBSCAN(eps=5.0, min_samples=5)
    labels_db = dbscan.fit_predict(features)
    # DBSCAN can produce -1 (noise). Handle for metrics?
    # For ARI/Silhouette, -1 is treated as a cluster or noise.
    if len(set(labels_db)) > 1:
        results['DBSCAN'] = evaluate_clustering(features, labels_db, labels_true)
    else:
        results['DBSCAN'] = {'Silhouette': -1, 'Davies-Bouldin': -1, 'ARI': -1, 'n_clusters': 1}

    return results, labels_kmeans # Return labels for visualization (using K-Means as representative)

def evaluate_clustering(X, labels, labels_true):
    # Skip if only 1 cluster
    if len(set(labels)) < 2:
         return {'Silhouette': -1, 'Davies-Bouldin': -1, 'ARI': -1, 'n_clusters': 1}
         
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ari = adjusted_rand_score(labels_true, labels)
    
    return {
        'Silhouette': sil,
        'Davies-Bouldin': db,
        'ARI': ari,
        'n_clusters': len(set(labels))
    }

def visualize_tsne(features, labels, title, filename):
    print(f"Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, palette='viridis', s=10)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def main():
    df, audio_feat, hybrid_feat = load_data_and_features()
    
    labels_true = df['language'].values
    
    # Compare Audio Only vs Hybrid
    print("\n--- Running Clustering on Audio-Only Features ---")
    results_audio, labels_audio = run_clustering(audio_feat, labels_true, "Audio")
    
    print("\n--- Running Clustering on Hybrid Features ---")
    results_hybrid, labels_hybrid = run_clustering(hybrid_feat, labels_true, "Hybrid")
    
    # Print Comparison Table
    print("\n" + "="*80)
    print(f"{'Method':<20} | {'Algorithm':<15} | {'Sil score':<10} | {'DB Index':<10} | {'ARI':<10}")
    print("-" * 80)
    
    for feat_type, results in [('Audio', results_audio), ('Hybrid', results_hybrid)]:
        for algo, metrics in results.items():
            print(f"{feat_type:<20} | {algo:<15} | {metrics['Silhouette']:<10.4f} | {metrics['Davies-Bouldin']:<10.4f} | {metrics['ARI']:<10.4f}")
    print("="*80)
    
    # Save Metrics to file
    with open("hybrid_metrics.txt", "w") as f:
         f.write(f"{'Method':<20} | {'Algorithm':<15} | {'Sil score':<10} | {'DB Index':<10} | {'ARI':<10}\n")
         f.write("-" * 80 + "\n")
         for feat_type, results in [('Audio', results_audio), ('Hybrid', results_hybrid)]:
            for algo, metrics in results.items():
                f.write(f"{feat_type:<20} | {algo:<15} | {metrics['Silhouette']:<10.4f} | {metrics['Davies-Bouldin']:<10.4f} | {metrics['ARI']:<10.4f}\n")

    # Visualizations
    visualize_tsne(audio_feat, df['language'], "t-SNE Audio Features (Ground Truth)", "tsne_audio_gt.png")
    visualize_tsne(audio_feat, labels_audio, "t-SNE Audio Features (K-Means)", "tsne_audio_kmeans.png")
    
    visualize_tsne(hybrid_feat, df['language'], "t-SNE Hybrid Features (Ground Truth)", "tsne_hybrid_gt.png")
    visualize_tsne(hybrid_feat, labels_hybrid, "t-SNE Hybrid Features (K-Means)", "tsne_hybrid_kmeans.png")
    
    print("Analysis Complete. Results saved.")

if __name__ == "__main__":
    main()
