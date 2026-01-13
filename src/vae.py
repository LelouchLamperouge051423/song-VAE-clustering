"""
VAE Model Definitions
Contains VAE, CVAE, Beta-VAE, and Autoencoder architectures
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """Convolutional Variational Autoencoder for audio features"""
    
    def __init__(self, input_dim=40, hidden_dim=64):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, 4, 2, 1),  # (64, 215)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, 2, 1),  # (128, 107)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, 2, 1),  # (256, 53)
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 53, hidden_dim)
        self.fc_logvar = nn.Linear(256 * 53, hidden_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(hidden_dim, 256 * 53)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1, output_padding=1),  # (128, 107)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),  # (64, 215)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, 4, 2, 1),  # (40, 430)
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 53)
        recon = self.decoder(h)
        
        return recon, mu, logvar


class CVAE(nn.Module):
    """Conditional VAE with language conditioning"""
    
    def __init__(self, input_dim=40, latent_dim=64, num_classes=2):
        super(CVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, 16)
        
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
        )
        
        # Latent (conditioned)
        self.fc_mu = nn.Linear(256 * 53 + 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 53 + 16, latent_dim)
        
        # Decoder (conditioned)
        self.fc_decode = nn.Linear(latent_dim + 16, 256 * 53)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, 4, 2, 1),
        )
    
    def encode(self, x, c):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        c_emb = self.label_emb(c)
        h = torch.cat([h, c_emb], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        c_emb = self.label_emb(c)
        h = torch.cat([z, c_emb], dim=1)
        h = self.fc_decode(h)
        h = h.view(h.size(0), 256, 53)
        return self.decoder(h)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar


class SimpleAutoencoder(nn.Module):
    """Standard Autoencoder without VAE regularization"""
    
    def __init__(self, input_dim=40, latent_dim=64):
        super(SimpleAutoencoder, self).__init__()
        
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
        )
        
        self.fc_encode = nn.Linear(256 * 53, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 53)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, 4, 2, 1),
        )
    
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.fc_encode(h)
        
        # Decode
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 53)
        recon = self.decoder(h)
        
        return recon, z


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function with adjustable beta for Beta-VAE"""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def ae_loss(recon_x, x):
    """Simple reconstruction loss for Autoencoder"""
    return nn.functional.mse_loss(recon_x, x, reduction='sum')
