# src/audio/augment.py
#
# Offline augmentation helpers used in prepare_data.py.
# augment_sample() produces n_augments copies (default 5 → ~6x total data).

import numpy as np
import librosa


# ── Individual transforms ─────────────────────────────────────────────────────

def add_noise(audio, noise_factor=None):
    if noise_factor is None:
        noise_factor = np.random.uniform(0.002, 0.008)
    return audio + noise_factor * np.random.randn(len(audio))


def time_shift(audio, shift_max=0.2):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(audio))
    return np.roll(audio, shift)


def pitch_shift(audio, sample_rate, pitch_steps=None):
    if pitch_steps is None:
        pitch_steps = np.random.uniform(-2, 2)
    try:
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_steps)
    except Exception:
        return audio


def time_stretch(audio, rate=None):
    if rate is None:
        rate = np.random.uniform(0.85, 1.15)
    try:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        if len(stretched) > len(audio):
            return stretched[:len(audio)]
        return np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
    except Exception:
        return audio


def change_volume(audio, gain=None):
    if gain is None:
        gain = np.random.uniform(0.7, 1.3)
    return audio * gain


def add_echo(audio, sr, delay_ms=None, decay=None):
    """Single-tap echo / lite-reverb."""
    if delay_ms is None:
        delay_ms = np.random.uniform(20, 60)
    if decay is None:
        decay = np.random.uniform(0.2, 0.4)
    delay_samples = int(sr * delay_ms / 1000)
    echo = np.zeros_like(audio)
    if delay_samples < len(audio):
        echo[delay_samples:] = audio[:-delay_samples] * decay
    return audio + echo


def freq_mask(audio, sr, num_masks=1):
    """Approximate frequency masking via STFT zero-out + ISTFT."""
    try:
        D      = librosa.stft(audio)
        n_freq = D.shape[0]
        for _ in range(num_masks):
            f0 = np.random.randint(0, n_freq - 10)
            f1 = f0 + np.random.randint(5, 15)
            D[f0:f1, :] = 0
        return librosa.istft(D, length=len(audio))
    except Exception:
        return audio


# ── Fear-specific augmentation ────────────────────────────────────────────────

def fear_variant(audio, sr, variant):
    """
    4 targeted fear augmentations to boost the minority class.
    Called from prepare_data.py only for the fear label.
    """
    audio = audio.copy()

    if variant == 0:
        # Anxious urgency: slight pitch-up + louder + light noise
        audio = pitch_shift(audio, sr, pitch_steps=np.random.uniform(2, 4))
        audio = audio * np.random.uniform(1.1, 1.3)
        audio = audio + np.random.randn(len(audio)) * 0.008

    elif variant == 1:
        # Frozen/shocked: slower + quieter + minimal noise
        audio = time_stretch(audio, rate=np.random.uniform(0.80, 0.90))
        audio = audio * np.random.uniform(0.75 , 0.90)
        audio = audio + np.random.randn(len(audio)) * 0.003

    elif variant == 2:
        # Panicked: faster + slight pitch-up + noise
        audio = time_stretch(audio, rate=np.random.uniform(1.1, 1.25))
        audio = pitch_shift(audio, sr, pitch_steps=np.random.uniform(1, 2))
        audio = audio + np.random.randn(len(audio)) * 0.005

    elif variant == 3:
        # Voice tremor / vibrato
        t      = np.linspace(0, len(audio) / sr, len(audio))
        tremor = 1.0 + np.random.uniform(0.1, 0.25) * \
                 np.sin(2 * np.pi * np.random.uniform(4, 8) * t)
        audio  = audio * tremor
        audio  = audio + np.random.randn(len(audio)) * 0.004

    return audio


# ── Main entry-point ─────────────────────────────────────────────────────────

def augment_sample(audio, sample_rate, n_augments=5):
    """
    Produce n_augments augmented copies of a raw waveform.
    Each copy randomly chains 2-3 transforms from the pool of 7.

    n_augments=5 → original + 5 copies ≈ 6x total data.
    """
    pool = [
        lambda a: add_noise(a),
        lambda a: time_shift(a),
        lambda a: pitch_shift(a, sample_rate),
        lambda a: time_stretch(a),
        lambda a: change_volume(a),
        lambda a: add_echo(a, sample_rate),
        lambda a: freq_mask(a, sample_rate),
    ]

    results = []
    for _ in range(n_augments):
        aug = audio.copy()
        n   = np.random.randint(2, 4)                         # 2 or 3 transforms
        for idx in np.random.choice(len(pool), size=n, replace=False):
            try:
                aug = pool[idx](aug)
            except Exception:
                pass
        results.append(aug)

    return results