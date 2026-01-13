import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import umap
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Config
DATA_DIR = Path("data")
INPUT_DIM = 40  # MFCC has 40 coefficients
SEQ_LEN = 430   # ~10 seconds
BATCH_SIZE = 64
LATENT_DIM = 64
EPOCHS = 1
# ...
if __name__ == "__main__":
    trained_model = train_model()
    # if trained_model:
    #     analyze_latent_space(trained_model)
LEARNING_RATE = 1e-4 # Reduced from 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATASET
# =========================
class SongDataset(Dataset):
    def __init__(self, root_dir, seq_len=SEQ_LEN):
        self.files = []
        self.labels = [] # 0 for english, 1 for bangla
        self.seq_len = seq_len
        
        # Load English
        eng_path = root_dir / "english" / "features"
        if eng_path.exists():
            eng_files = list(eng_path.glob("*.npy"))
            self.files.extend(eng_files)
            self.labels.extend([0] * len(eng_files))
        
        # Load Bangla
        ban_path = root_dir / "bangla" / "features"
        if ban_path.exists():
            ban_files = list(ban_path.glob("*.npy"))
            self.files.extend(ban_files)
            self.labels.extend([1] * len(ban_files))
        
        logger.info(f"Found {len(eng_files) if eng_path.exists() else 0} English and {len(ban_files) if ban_path.exists() else 0} Bangla segments.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            mfcc = np.load(path) # (40, T)
        except:
            # Fallback for corrupt files
            mfcc = np.zeros((INPUT_DIM, self.seq_len))
        
        # 1. Pad or Crop
        c, t = mfcc.shape
        if t < self.seq_len:
            pad_width = self.seq_len - t
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.seq_len]
            
        # 2. Per-Instance Normalization (Z-score)
        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True) + 1e-6
        mfcc = (mfcc - mean) / std
            
        return torch.FloatTensor(mfcc), self.labels[idx]

# =========================
# VAE MODEL
# =========================
class VAE(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=LATENT_DIM):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, 4, 2, 1), # (64, 215)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 1),       # (128, 107)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, 2, 1),      # (256, 53)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.flat_dim = 256 * 53
        
        self.fc_mu = nn.Linear(self.flat_dim, hidden_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, hidden_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(hidden_dim, self.flat_dim)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 53)),
            nn.ConvTranspose1d(256, 128, 4, 2, 1, output_padding=1), # (128, 107)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1, output_padding=1),  # (64, 215)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, 4, 2, 1),              # (40, 430)
            # No BatchNorm/ReLU at output, raw logits/values
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(self.decoder_input(z))
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Scale KLD ? Sometimes useful, but let's stick to standard ELBO
    return mse + kld

# =========================
# TRAINING
# =========================
def train_model():
    dataset = SongDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info(f"Starting training on {DEVICE} for {EPOCHS} epochs...")
    model.train()
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(DEVICE)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar)
            
            if torch.isnan(loss):
                logger.error("Loss is NaN!")
                continue
                
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), "vae_model.pth")
    
    plt.figure()
    plt.plot(loss_history)
    plt.title("VAE Training Loss")
    plt.savefig("vae_loss.png")
    
    return model

# =========================
# ANALYSIS
# =========================
def analyze_latent_space(model):
    logger.info("Extracting latent features...")
    dataset = SongDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(DEVICE)
            _, mu, _ = model(data)
            latents.append(mu.cpu().numpy())
            labels.append(label.numpy())
            
    latents = np.concatenate(latents, axis=0) # (N, 64)
    labels = np.concatenate(labels, axis=0)   # (N,)
    
    # K-Means
    logger.info("Clustering...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(latents)
    
    sil = silhouette_score(latents, clusters)
    ch = calinski_harabasz_score(latents, clusters)
    
    logger.info(f"VAE | Silhouette: {sil:.4f} | Calinski-Harabasz: {ch:.4f}")
    
    # t-SNE
    logger.info("t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    z_emb = tsne.fit_transform(latents)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    # Map 0->Eng, 1->Ban
    sns.scatterplot(x=z_emb[:,0], y=z_emb[:,1], hue=labels, palette={0: 'blue', 1: 'red'}, alpha=0.5)
    plt.title("Ground Truth (Blue=Eng, Red=Ban)")
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=z_emb[:,0], y=z_emb[:,1], hue=clusters, palette="viridis", alpha=0.5)
    plt.title("K-Means Clusters")
    
    plt.savefig("cluster_analysis.png")
    logger.info("Saved plots.")

if __name__ == "__main__":
    trained_model = train_model()
    if trained_model:
        analyze_latent_space(trained_model)
