"""
Configuration file for hyperparameters and paths
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
VIZ_DIR = RESULTS_DIR / "latent_visualization"

# Model hyperparameters
INPUT_DIM = 40  # Number of MFCC coefficients
SEQ_LEN = 430   # Sequence length (time frames)
LATENT_DIM = 64  # Latent space dimension

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
DEVICE = "cuda"  # or "cpu"

# VAE hyperparameters
BETA_VAE = 4.0  # Beta for Beta-VAE
NUM_CLASSES = 2  # English vs Bangla

# Clustering hyperparameters
K_CLUSTERS = 2  # Number of clusters
RANDOM_SEED = 42

# Multi-modal feature dimensions
TEXT_DIM = 32  # Dimension after PCA
ARTIST_DIM = 20  # Number of top artists to encode

# DBSCAN parameters
DBSCAN_EPS = 5.0
DBSCAN_MIN_SAMPLES = 5

# File paths
DATASET_INDEX = PROJECT_ROOT / "outputs" / "dataset_index.csv"

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)
