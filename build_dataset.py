import csv
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import yt_dlp

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUTS = PROJECT_ROOT / "outputs"
CSV_PATH = OUTPUTS / "dataset_index.csv"
SOURCES_CSV = PROJECT_ROOT / "youtube_sources.csv"

SAMPLE_RATE = 22050
SEGMENT_DURATION = 30  # seconds

LANGUAGES = {
    "english": {
        "folder": "english",
        "target": 300
    },
    "bangla": {
        "folder": "bangla",
        "target": 300
    }
}

# =========================
# LOGGING
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("dataset_builder")

# =========================
# DIRECTORY SETUP
# =========================

def setup_dirs():
    for lang in LANGUAGES.values():
        base = DATA_ROOT / lang["folder"]
        (base / "audio").mkdir(parents=True, exist_ok=True)
        (base / "lyrics").mkdir(parents=True, exist_ok=True)
        (base / "features").mkdir(parents=True, exist_ok=True)
        (base / "segments").mkdir(parents=True, exist_ok=True) # New folder for segments

    OUTPUTS.mkdir(exist_ok=True)

# =========================
# YOUTUBE AUDIO DOWNLOAD
# =========================

def download_audio(url: str, out_path: Path) -> bool:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_path),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav"
        }],
        "quiet": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Add strict duration filter (SKIP if > 6 mins)
            info = ydl.extract_info(url, download=False)
            if 'entries' in info:
                info = info['entries'][0]
            
            duration = info.get('duration', 0)
            if duration > 360: # 6 minutes
                logger.warning(f"Skipping {url}: Duration {duration}s > 360s")
                return False
                
            ydl.download([url])
        
        # Check if file exists (yt-dlp might append .wav automatically)
        expected_path = out_path.with_suffix(".wav")
        return expected_path.exists()
    except Exception as e:
        logger.warning(f"Download failed for {url}: {e}")
        return False

# =========================
# AUDIO SEGMENTATION
# =========================

def segment_audio(audio_path: Path, out_dir: Path, base_id: str):
    """
    Splits the audio file into 30s segments.
    Returns a list of paths to the segments.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        duration = librosa.get_duration(y=y, sr=sr)
        
        segments = []
        count = int(duration // SEGMENT_DURATION)
        
        # If less than one segment, we skip (or handle otherwise)
        if count == 0:
             return []

        for i in range(count):
            start = int(i * SEGMENT_DURATION * sr)
            end = int((i + 1) * SEGMENT_DURATION * sr)
            seg = y[start:end]

            seg_fname = out_dir / f"{base_id}_seg{i}.wav"
            sf.write(seg_fname, seg, sr)
            segments.append(seg_fname)

        return segments
    except Exception as e:
        logger.error(f"Segmentation failed for {audio_path}: {e}")
        return []

# =========================
# FEATURE EXTRACTION (VAE INPUT)
# =========================

def extract_mfcc(audio_path: Path, out_path: Path) -> bool:
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        # Ensure consistent length if needed
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        np.save(out_path, mfcc)
        return True
    except Exception as e:
        logger.error(f"Feature extraction failed for {audio_path}: {e}")
        return False

# =========================
# DATA LOADING
# =========================

def load_sources_from_csv(csv_path: Path):
    sources = {"english": [], "bangla": []}
    
    if not csv_path.exists():
        logger.error(f"Sources file not found: {csv_path}")
        return sources

    try:
        df = pd.read_csv(csv_path)
        # Clean header nicely
        df.columns = [c.strip().lower() for c in df.columns]
        
        for _, row in df.iterrows():
            lang = row.get("language", "").strip().lower()
            url = row.get("youtube_url", "").strip()
            
            # Skip invalid or placeholder URLs
            # Allow http links AND ytsearch: queries
            if not url or ("http" not in url and "ytsearch" not in url):
                continue
                
            if lang in sources:
                sources[lang].append({
                    "youtube": url,
                    "lyrics": row.get("lyrics", ""),
                    "title": row.get("title", ""),
                    "artist": row.get("artist", "")
                })
                
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        
    return sources

# =========================
# MAIN PIPELINE
# =========================

def main():
    setup_dirs()
    
    logger.info("Loading sources from CSV...")
    song_sources = load_sources_from_csv(SOURCES_CSV)
    
    rows = []

    for lang, songs in song_sources.items():
        if not songs:
            logger.info(f"No valid songs found for '{lang}'. Please check {SOURCES_CSV}")
            continue

        folder = LANGUAGES[lang]["folder"]
        base = DATA_ROOT / folder
        
        logger.info(f"Processing {len(songs)} songs for {lang}...")

        for idx, song in enumerate(tqdm(songs, desc=f"{lang} songs"), 1):
            base_id = f"{folder}_{idx:04d}"
            
            # Directories
            audio_dir = base / "audio"
            lyrics_dir = base / "lyrics"
            features_dir = base / "features" 
            segments_dir = base / "segments" 

            audio_file_base = audio_dir / base_id # without extension
            
            # 1. Download
            if not download_audio(song["youtube"], audio_file_base):
                continue

            full_audio_path = audio_file_base.with_suffix(".wav")
            
            # 2. Save Lyrics (One file per song, could be multiple segments pointing to it)
            lyrics_path = lyrics_dir / f"{base_id}.txt"
            lyrics_text = str(song["lyrics"]) if pd.notna(song["lyrics"]) else "Lyrics not available"
            lyrics_path.write_text(lyrics_text, encoding="utf-8", errors='ignore')

            # 3. Segment Audio
            segment_paths = segment_audio(full_audio_path, segments_dir, base_id)
            
            if not segment_paths:
                logger.warning(f"No valid segments for {base_id} (might be too short)")
                continue

            # 4. Extract Features for EACH segment
            for seg_idx, seg_path in enumerate(segment_paths):
                seg_id = f"{base_id}_seg{seg_idx}"
                feature_path = features_dir / f"{seg_id}.npy"
                
                if extract_mfcc(seg_path, feature_path):
                    rows.append({
                        "id": seg_id,
                        "parent_id": base_id,
                        "language": folder,
                        "segment_audio": str(seg_path),
                        "full_audio": str(full_audio_path),
                        "lyrics_file": str(lyrics_path),
                        "features": str(feature_path),
                        "youtube": song["youtube"],
                        "title": song["title"],
                        "artist": song["artist"]
                    })

    # Save Index
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(CSV_PATH, index=False)
        logger.info(f"Dataset generation complete. Index saved to {CSV_PATH}")
        logger.info(f"Total Segments: {len(df)}")
    else:
        logger.warning(f"No data generated. Ensure '{SOURCES_CSV}' has valid links (not just placeholders).")

if __name__ == "__main__":
    main()
