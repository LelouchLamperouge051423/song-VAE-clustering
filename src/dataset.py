"""
Dataset classes for audio feature loading
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd


class SongDataset(Dataset):
    """Dataset for loading MFCC features from audio segments"""
    
    def __init__(self, root_dir, seq_len=430):
        self.files = []
        self.labels = []  # 0 for english, 1 for bangla
        self.seq_len = seq_len
        
        root_path = Path(root_dir)
        
        # Load English
        eng_path = root_path / "english" / "features"
        if eng_path.exists():
            eng_files = sorted(eng_path.glob("*.npy"))
            self.files.extend(eng_files)
            self.labels.extend([0] * len(eng_files))
        
        # Load Bangla
        ban_path = root_path / "bangla" / "features"
        if ban_path.exists():
            ban_files = sorted(ban_path.glob("*.npy"))
            self.files.extend(ban_files)
            self.labels.extend([1] * len(ban_files))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load MFCC
        mfcc = np.load(self.files[idx])  # Shape: (n_mfcc, T)
        
        # Pad or truncate
        if mfcc.shape[1] < self.seq_len:
            pad_width = self.seq_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :self.seq_len]
        
        # Normalize
        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True) + 1e-8
        mfcc = (mfcc - mean) / std
        
        return torch.FloatTensor(mfcc), self.labels[idx]


class SongDatasetWithLabels(Dataset):
    """Dataset that includes language labels for CVAE conditioning"""
    
    def __init__(self, csv_path, seq_len=430):
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.valid_indices = []
        
        # Filter valid samples
        for idx, row in self.df.iterrows():
            feat_path = Path(row['features'])
            if feat_path.exists():
                self.valid_indices.append(idx)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        
        # Load MFCC
        mfcc = np.load(row['features'])
        
        # Pad or truncate
        if mfcc.shape[1] < self.seq_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, self.seq_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :self.seq_len]
        
        # Normalize
        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True) + 1e-8
        mfcc = (mfcc - mean) / std
        
        # Get label
        language = row['language'].strip().lower()
        label = 0 if language == 'english' else 1
        
        return torch.FloatTensor(mfcc), label
