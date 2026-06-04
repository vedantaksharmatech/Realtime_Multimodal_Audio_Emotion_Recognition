# src/audio/prepare_data.py
#
# CREMA-D ONLY.
# 5x offline augmentation on all training samples (≈ 6x total).
# Fear class gets 4 extra fear-specific variants on top.
# Calibration techniques COMMENTED OUT — not deleted.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.audio.build_dataset import build_audio_dataset
from src.audio.augment import augment_sample, fear_variant
import librosa
import pickle
import os
import matplotlib.pyplot as plt
import librosa.display

# ── MFCC image helper ─────────────────────────────────────────────────────────
def save_mfcc_image(mfcc, save_path):
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title("MFCC Spectrogram")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_PAD_LEN = 174
N_MFCC      = 40
N_AUGMENTS  = 5        # 5 offline copies per sample → ~6x total

image_folder = "models/mfcc_images"
os.makedirs(image_folder, exist_ok=True)
image_count = 0
MAX_IMAGES  = 200

# ── MFCC extraction ───────────────────────────────────────────────────────────
def _mfcc(audio, sr):
    m = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    if m.shape[1] < MAX_PAD_LEN:
        m = np.pad(m, ((0, 0), (0, MAX_PAD_LEN - m.shape[1])), mode='constant')
    else:
        m = m[:, :MAX_PAD_LEN]
    return m   # (40, 174)

# ── Main pipeline ─────────────────────────────────────────────────────────────
def prepare_data(cremad_path="data/CREMAD"):

    global image_count

    # 1. Load raw audio (CREMA-D only)
    raw_samples = build_audio_dataset(cremad_path=cremad_path)
    n_total     = len(raw_samples)
    print(f"\nExtracting MFCC from {n_total} original samples...")

    feats, labels = [], []

    # 2. Extract MFCC + optionally save images
    for i, (audio, sr, label) in enumerate(raw_samples):
        try:
            mfcc = _mfcc(audio, sr)
            feats.append(mfcc)
            labels.append(label)

            if image_count < MAX_IMAGES:
                save_mfcc_image(mfcc, os.path.join(image_folder, f"{label}_{i}.png"))
                image_count += 1

        except Exception as e:
            print(f"  Skip {i}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n_total}...")

    X     = np.array(feats)[..., np.newaxis]   # (N, 40, 174, 1)
    y_str = np.array(labels)

    # 3. Label encoding
    le = LabelEncoder()
    y  = le.fit_transform(y_str)

    print("\nLabel encoding:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i} → {cls}")

    # 4. Train / Test split (stratified)
    X_train, X_test, y_train, y_test, idx_train, _ = train_test_split(
        X, y, np.arange(n_total),
        test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape}")
    print(f"Test : {X_test.shape}")

    # 5. Offline augmentation (5x on ALL training samples)
    print(f"\nAugmenting training set ({N_AUGMENTS}x per sample)...")

    fear_idx = int(np.where(le.classes_ == "fear")[0][0]) \
               if "fear" in le.classes_ else -1

    aug_X, aug_y = [], []

    for pos, orig_idx in enumerate(idx_train):
        audio, sr, label = raw_samples[orig_idx]
        label_enc = y[orig_idx]

        # Standard 5x augmentation for every sample
        for aug_audio in augment_sample(audio, sr, n_augments=N_AUGMENTS):
            try:
                aug_X.append(_mfcc(aug_audio, sr)[..., np.newaxis])
                aug_y.append(label_enc)
            except Exception:
                pass

        # Extra 4 fear-specific variants for fear class
        if label_enc == fear_idx:
            for variant in range(4):
                try:
                    aug_X.append(
                        _mfcc(fear_variant(audio, sr, variant), sr)[..., np.newaxis]
                    )
                    aug_y.append(label_enc)
                except Exception:
                    pass

        if (pos + 1) % 500 == 0:
            print(f"  Augmented {pos+1}/{len(idx_train)} train samples...")

    if aug_X:
        X_train = np.concatenate([X_train, np.array(aug_X)], axis=0)
        y_train = np.concatenate([y_train, np.array(aug_y)], axis=0)

    print(f"Training after augmentation: {X_train.shape}")

    # 6. Normalisation (z-score, clipped)
    print("\nNormalising...")
    mean = float(np.mean(X_train))
    std  = float(np.std(X_train))
    X_train = np.clip((X_train - mean) / (std + 1e-8), -5, 5)
    X_test  = np.clip((X_test  - mean) / (std + 1e-8), -5, 5)

    # 7. Class-balance via undersampling
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nPre-undersample class counts:")
    for u, c in zip(unique, counts):
        print(f"  {le.classes_[u]}: {c}")

    #minority = int(np.percentile(counts,25))   
    minority = int(np.min(counts))
    bal_idx  = []
    for cls in np.unique(y_train):
        idx    = np.where(y_train == cls)[0]
        chosen = np.random.choice(idx, size=minority, replace=False)
        bal_idx.extend(chosen.tolist())

    np.random.shuffle(bal_idx)
    X_train = X_train[bal_idx]
    y_train = y_train[bal_idx]
    print(f"Final training set (balanced): {X_train.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # CALIBRATION — commented out, NOT deleted.
    # ─────────────────────────────────────────────────────────────────────────
    # from sklearn.calibration import CalibratedClassifierCV
    # calibrated_model = CalibratedClassifierCV(base_estimator=model,
    #                                           method='sigmoid', cv='prefit')
    # calibrated_model.fit(X_val, y_val)
    # prediction = calibrated_model.predict_proba(features)
    # ─────────────────────────────────────────────────────────────────────────

    # 8. Save artefacts
    os.makedirs("models", exist_ok=True)

    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    with open("models/norm_stats.pkl", "wb") as f:
        pickle.dump({"mean": mean, "std": std}, f)

    np.save("models/X_train.npy", X_train)
    np.save("models/X_test.npy",  X_test)
    np.save("models/y_train.npy", y_train)
    np.save("models/y_test.npy",  y_test)

    print("\n✅ Data preparation complete!")
    print(f"   MFCC images saved : {image_count}")
    print(f"   Final train shape : {X_train.shape}")
    print(f"   Final test  shape : {X_test.shape}")


if __name__ == "__main__":
    prepare_data()