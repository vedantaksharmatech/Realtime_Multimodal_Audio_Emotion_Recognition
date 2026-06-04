# src/audio/build_dataset.py
#
# CREMA-D ONLY — RAVDESS removed.
# Returns RAW waveforms only (no features, no augmentation).
# Both happen in prepare_data.py AFTER the train/test split.

import os
import numpy as np
import librosa
from collections import Counter

TARGET_EMOTIONS = ["angry", "fear", "happy", "sad"]

CREMAD_EMOTION_MAP = {
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fear",
    "DIS": "sad",     # disgust → sad (closest mapping)
}


def _load_cremad_raw(data_path):
    samples, skipped = [], 0
    if not os.path.exists(data_path):
        print(f"[CREMA-D] Path not found: {data_path} — skipping.")
        return samples
    for file in sorted(os.listdir(data_path)):
        if not file.endswith(".wav"):
            continue
        parts = file.replace(".wav", "").split("_")
        if len(parts) < 3:
            continue
        code = parts[2].upper()
        if code not in CREMAD_EMOTION_MAP:
            skipped += 1
            continue
        try:
            audio, sr = librosa.load(os.path.join(data_path, file), sr=16000)
            samples.append((audio, sr, CREMAD_EMOTION_MAP[code]))
        except Exception as e:
            print(f"[CREMA-D] Load error {file}: {e}")
    print(f"[CREMA-D] Loaded {len(samples)} | Skipped {skipped} (neutral/other)")
    return samples


def build_audio_dataset(cremad_path="data/CREMAD"):
    """
    Returns raw waveforms as list of (audio, sr, label) tuples.
    CREMA-D only. Feature extraction and augmentation happen in prepare_data.py.
    """
    print("=" * 60)
    print("Loading raw audio — CREMA-D ONLY")
    print("=" * 60)

    all_samples = _load_cremad_raw(cremad_path)

    print(f"\nTotal raw samples: {len(all_samples)}")
    print("Emotion counts (before augmentation):")
    counts = Counter(s[2] for s in all_samples)
    for e in TARGET_EMOTIONS:
        print(f"  {e}: {counts.get(e, 0)}")
    print("=" * 60)

    return all_samples