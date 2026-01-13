"""
Advanced VAE Training Script
- Conditional VAE (CVAE) with language conditioning
- Beta-VAE for disentangled representations
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Config
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_INDEX = PROJECT_ROOT / "outputs" / "dataset_index.csv"
INPUT_DIM = 40  # MFCC coefficients
SEQ_LEN = 430   # ~10 seconds
BATCH_SIZE = 64
LATENT_DIM = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 2  # English, Bangla
BETA = 4.0  # Beta-VAE weight
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATASET
# =========================
class SongDatasetWithLabels(Dataset):
    """Dataset that includes language labels for CVAE conditioning"""
    def __init__(self, csv_path, seq_len=SEQ_LEN):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.language_map = {'english': 0, 'bangla': 1}
        
        # Filter valid entries
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            feat_path = Path(row['features'])
            if feat_path.exists():
                self.valid_indices.append(idx)
        
        logger.info(f"Found {len(self.valid_indices)} valid samples")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        row = self.df.iloc[self.valid_indices[idx]]
        feat_path = Path(row['features'])
        
        try:
            mfcc = np.load(feat_path)
        except:
            mfcc = np.zeros((INPUT_DIM, self.seq_len))
        
        # Validate shape
        if mfcc.ndim != 2 or mfcc.shape[0] != INPUT_DIM:
            mfcc = np.zeros((INPUT_DIM, self.seq_len))
        
        # Pad or crop
        if mfcc.shape[1] < self.seq_len:
            pad_width = self.seq_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.seq_len]
        
        # Normalize
        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True) + 1e-6
        mfcc = (mfcc - mean) / std
        
        # Get language label
        lang = row['language'].strip().lower()
        label = self.language_map.get(lang, 0)
        
        return torch.FloatTensor(mfcc), label

# =========================
# CVAE MODEL
# =========================
class CVAE(nn.Module):
    """Conditional VAE with language conditioning"""
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super(CVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Condition embedding
        self.cond_embed = nn.Embedding(num_classes, 32)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, 2, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.flat_dim = 256 * 53
        
        # Condition is concatenated before FC layers
        self.fc_mu = nn.Linear(self.flat_dim + 32, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim + 32, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim + 32, self.flat_dim)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 53)),
            nn.ConvTranspose1d(256, 128, 4, 2, 1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, 4, 2, 1),
        )
    
    def encode(self, x, c):
        h = self.encoder(x)
        c_emb = self.cond_embed(c)  # (B, 32)
        h_c = torch.cat([h, c_emb], dim=1)
        return self.fc_mu(h_c), self.fc_logvar(h_c)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        c_emb = self.cond_embed(c)
        z_c = torch.cat([z, c_emb], dim=1)
        h = self.decoder_input(z_c)
        return self.decoder(h)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

# =========================
# SIMPLE AUTOENCODER (Baseline)
# =========================
class SimpleAutoencoder(nn.Module):
    """Standard Autoencoder without VAE regularization for baseline comparison"""
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super(SimpleAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, 2, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 53, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 53),
            nn.Unflatten(1, (256, 53)),
            nn.ConvTranspose1d(256, 128, 4, 2, 1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, 4, 2, 1),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# =========================
# LOSS FUNCTIONS
# =========================
def cvae_loss(recon_x, x, mu, logvar, beta=1.0):
    """CVAE/Beta-VAE loss with adjustable beta"""
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kld

def ae_loss(recon_x, x):
    """Simple reconstruction loss for Autoencoder"""
    return nn.functional.mse_loss(recon_x, x, reduction='sum')

# =========================
# TRAINING
# =========================
def train_cvae(beta=1.0, model_name="cvae"):
    """Train CVAE or Beta-VAE (beta > 1)"""
    dataset = SongDatasetWithLabels(DATASET_INDEX)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = CVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info(f"Training {model_name} (beta={beta}) on {DEVICE} for {EPOCHS} epochs...")
    model.train()
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for data, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(data, labels)
            loss = cvae_loss(recon, data, mu, logvar, beta=beta)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Save model
    save_path = f"{model_name}_model.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    
    # Plot loss
    plt.figure()
    plt.plot(loss_history)
    plt.title(f"{model_name.upper()} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{model_name}_loss.png")
    plt.close()
    
    return model

def train_autoencoder():
    """Train simple autoencoder baseline"""
    dataset = SongDatasetWithLabels(DATASET_INDEX)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = SimpleAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    logger.info(f"Training Autoencoder on {DEVICE} for {EPOCHS} epochs...")
    model.train()
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for data, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            data = data.to(DEVICE)
            
            optimizer.zero_grad()
            recon, _ = model(data)
            loss = ae_loss(recon, data)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "autoencoder_model.pth")
    
    plt.figure()
    plt.plot(loss_history)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("autoencoder_loss.png")
    plt.close()
    
    return model

if __name__ == "__main__":
    # Train CVAE (beta=1)
    logger.info("=" * 50)
    logger.info("Training CVAE (beta=1)")
    train_cvae(beta=1.0, model_name="cvae")
    
    # Train Beta-VAE (beta=4)
    logger.info("=" * 50)
    logger.info("Training Beta-VAE (beta=4)")
    train_cvae(beta=BETA, model_name="beta_vae")
    
    # Train Autoencoder baseline
    logger.info("=" * 50)
    logger.info("Training Autoencoder Baseline")
    train_autoencoder()
    
    logger.info("All models trained successfully!")
