import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import logging
from train_vae import VAE, SongDataset, DataLoader, DATA_DIR, DEVICE, BATCH_SIZE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def analyze_latent_space():
    logger.info("Loading model...")
    model = VAE().to(DEVICE)
    model.load_state_dict(torch.load("vae_model.pth", map_location=DEVICE))
    model.eval()
    
    logger.info("Extracting latent features...")
    dataset = SongDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(DEVICE)
            _, mu, _ = model(data)
            latents.append(mu.cpu().numpy())
            labels.append(label.numpy())
            
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # 1. K-Means
    logger.info("Clustering (K-Means)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(latents)
    
    sil = silhouette_score(latents, clusters)
    ch = calinski_harabasz_score(latents, clusters)
    
    # 1.5 Baseline: PCA on raw features
    logger.info("Computing Baseline (PCA + K-Means)...")
    # Flatten raw data for PCA: (N, 40, 430) -> (N, 17200)
    # We need to reload data or just reshape if we kept it? 
    # Dataset loader returns (B, 40, 430). We need the full raw array.
    # Re-iterating dataloader to get raw data
    raw_data = []
    for data, _ in dataloader:
        raw_data.append(data.view(data.size(0), -1).numpy())
    raw_data = np.concatenate(raw_data, axis=0)
    
    # PCA to same dim as latent (64) for fair comparison
    pca = PCA(n_components=64)
    raw_embedded = pca.fit_transform(raw_data)
    
    kmeans_base = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters_base = kmeans_base.fit_predict(raw_embedded)
    
    sil_base = silhouette_score(raw_embedded, clusters_base)
    ch_base = calinski_harabasz_score(raw_embedded, clusters_base)
    
    # Print to console
    print(f"\n=== RESULTS COMPARISON ===")
    print(f"{'Metric':<25} | {'VAE (Latent)':<15} | {'Baseline (PCA)':<15}")
    print(f"-"*60)
    print(f"{'Silhouette Score':<25} | {sil:<15.4f} | {sil_base:<15.4f}")
    print(f"{'Calinski-Harabasz':<25} | {ch:<15.4f} | {ch_base:<15.4f}")
    print(f"-"*60)
    
    # Save to file
    with open("metrics.txt", "w") as f:
        f.write(f"=== RESULTS COMPARISON ===\n")
        f.write(f"{'Metric':<25} | {'VAE (Latent)':<15} | {'Baseline (PCA)':<15}\n")
        f.write(f"-"*60 + "\n")
        f.write(f"{'Silhouette Score':<25} | {sil:<15.4f} | {sil_base:<15.4f}\n")
        f.write(f"{'Calinski-Harabasz':<25} | {ch:<15.4f} | {ch_base:<15.4f}\n")
        f.write(f"-"*60 + "\n")
    
    # 2. t-SNE
    logger.info("Generating t-SNE plot...")
    # Reduce dimensions for speed if needed, but 64 is small enough
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    z_emb = tsne.fit_transform(latents)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=z_emb[:,0], y=z_emb[:,1], hue=labels, palette={0: 'blue', 1: 'red'}, alpha=0.5, s=15)
    plt.title("Ground Truth (Blue=Eng, Red=Ban)")
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=z_emb[:,0], y=z_emb[:,1], hue=clusters, palette="viridis", alpha=0.5, s=15)
    plt.title("K-Means Clusters")
    
    plt.tight_layout()
    plt.savefig("cluster_analysis.png")
    logger.info("Saved cluster_analysis.png")

if __name__ == "__main__":
    analyze_latent_space()
