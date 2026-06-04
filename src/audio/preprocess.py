# src/audio/preprocess.py
#
# Pure MFCC extraction — 40 rows x 174 time frames.
# Used at inference time (file path) and during dataset building (raw array).

import librosa
import numpy as np

MAX_PAD_LEN = 174
N_MFCC      = 40


def extract_features(file_path):
    """
    Extract MFCC from a .wav file path.
    Returns shape (40, 174) or None on error.
    Used at inference time.
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return _extract(audio, sr)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_features_from_audio(audio, sample_rate):
    """
    Extract MFCC from a raw waveform array.
    Returns shape (40, 174) or None on error.
    Used during dataset building after augmentation.
    """
    try:
        return _extract(audio, sample_rate)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


def _extract(audio, sr, max_pad_len=MAX_PAD_LEN):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] < max_pad_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_len - mfcc.shape[1])),
                      mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc   # (40, 174)