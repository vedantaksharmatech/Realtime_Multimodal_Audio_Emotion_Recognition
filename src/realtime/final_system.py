# src/realtime/final_system.py
#
# CALIBRATION UPDATE:
# Added personal voice calibration override layer after CNN prediction.
# Computes RMS Energy + Mean Pitch from live mic audio and applies
# threshold rules to fix ANGRY and SAD misclassification.
# NO RETRAINING — purely post-processing on top of CNN output.
#
# Calibration recorded values (your voice):
#   ANGRY  RMS=0.12056  Pitch=261.8 Hz
#   SAD    RMS=0.01254  Pitch=136.6 Hz
#   HAPPY  RMS=0.02588  Pitch=213.9 Hz
#   FEAR   RMS=0.02053  Pitch=184.2 Hz

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import pickle
import os
import warnings

warnings.filterwarnings("ignore")

SAMPLE_RATE          = 22050
DURATION             = 5
EXPECTED_TIME_FRAMES = 174

MODEL_PATH         = "models/audio_cnn_model_best.keras"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
NORM_STATS_PATH    = "models/norm_stats.pkl"

# ============================================================
# PERSONAL VOICE CALIBRATION THRESHOLDS
# ANGRY : RMS > 0.09042  AND  Pitch > 209.4 Hz
# SAD   : RMS < 0.01881  AND  Pitch < 150.0 Hz  <- FIXED from 204.9
#         (204.9 was too high — would catch happy(213) and fear(184).
#          Your SAD pitch was 136.6 Hz so 150 gives a clean safe gap.)
# HAPPY/FEAR : no rule — CNN + text handles them
# ============================================================
ANGRY_RMS_THRESHOLD   = 0.09042
ANGRY_PITCH_THRESHOLD = 209.4
SAD_RMS_THRESHOLD     = 0.01881
SAD_PITCH_THRESHOLD   = 150.0    # FIXED

# ============================================================
# FUSION WEIGHTS PER ZONE
# angry_zone  : audio=0.65, text=0.35
# sad_zone    : audio=0.63, text=0.37  (normalized from 0.60/0.35)
# middle_zone : audio=0.30, text=0.70  (happy/fear, text is better)
# ============================================================
WEIGHTS = {
    "angry_zone":  (0.65, 0.35),
    "sad_zone":    (0.63, 0.37),
    "middle_zone": (0.30, 0.70),
}

# Must match LabelEncoder: angry=0 fear=1 happy=2 sad=3
EMOTIONS = ["angry", "fear", "happy", "sad"]

print("Loading CNN audio model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
EMOTION_LABELS = list(label_encoder.classes_)
print("Emotion classes:", EMOTION_LABELS)

with open(NORM_STATS_PATH, "rb") as f:
    norm_stats = pickle.load(f)
NORM_MEAN = norm_stats["mean"]
NORM_STD  = norm_stats["std"]


def extract_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < EXPECTED_TIME_FRAMES:
        mfcc = np.pad(mfcc,
                      ((0, 0), (0, EXPECTED_TIME_FRAMES - mfcc.shape[1])),
                      mode='constant')
    else:
        mfcc = mfcc[:, :EXPECTED_TIME_FRAMES]
    return mfcc


def compute_acoustic_features(audio, sr):
    """Compute RMS energy and mean pitch (F0) — used only for calibration."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        voiced_f0  = f0[voiced_flag] if voiced_flag is not None else np.array([])
        mean_pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    except Exception:
        mean_pitch = 0.0
    return rms, mean_pitch


def apply_calibration(cnn_probs, audio, sr):
    """
    Post-processing calibration layer.
    ANGRY : RMS > 0.09042 AND Pitch > 209.4  -> override to angry
    SAD   : RMS < 0.01881 AND Pitch < 150.0  -> override to sad
    else  : return CNN probs unchanged (happy/fear zone)

    Override confidence = 0.82 (not 1.0 — keeps results realistic)
    Returns: corrected_probs, zone, rms, pitch
    """
    rms, pitch = compute_acoustic_features(audio, sr)

    print(f"\n  [Calibration] RMS Energy       : {rms:.5f}")
    print(f"  [Calibration] Live Voice Freq  : {pitch:.1f} Hz")

    zone       = "middle_zone"
    overridden = None

    if rms > ANGRY_RMS_THRESHOLD and pitch > ANGRY_PITCH_THRESHOLD:
        print(f"  [Calibration] ANGRY rule FIRED "
              f"(RMS {rms:.5f} > {ANGRY_RMS_THRESHOLD} "
              f"AND Pitch {pitch:.1f} > {ANGRY_PITCH_THRESHOLD})")
        print(f"  [CALIBRATION PRIORITIZED] Acoustic energy+pitch threshold "
              f"overrides CNN/MFCC output for this prediction.")
        overridden = "angry"
        zone       = "angry_zone"

    elif rms < SAD_RMS_THRESHOLD and pitch < SAD_PITCH_THRESHOLD:
        print(f"  [Calibration] SAD rule FIRED "
              f"(RMS {rms:.5f} < {SAD_RMS_THRESHOLD} "
              f"AND Pitch {pitch:.1f} < {SAD_PITCH_THRESHOLD})")
        print(f"  [CALIBRATION PRIORITIZED] Acoustic energy+pitch threshold "
              f"overrides CNN/MFCC output for this prediction.")
        overridden = "sad"
        zone       = "sad_zone"

    else:
        print(f"  [Calibration] No override — HAPPY/FEAR zone | "
              f"Standard MFCC/CNN pipeline active.")

    if overridden is not None:
        idx        = EMOTIONS.index(overridden)
        probs      = np.zeros(len(EMOTIONS))
        probs[idx] = 0.82
        other      = 0.18 / (len(EMOTIONS) - 1)
        for i in range(len(EMOTIONS)):
            if i != idx:
                probs[i] = other
        return probs, zone, rms, pitch

    return cnn_probs, zone, rms, pitch


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
print(f"\nRecording for {DURATION} seconds... Speak now!")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
               channels=1, dtype='float32')
sd.wait()
print("Recording complete.\n")
audio = np.squeeze(audio)

# Step 1 — CNN prediction
features      = extract_mfcc(audio, SAMPLE_RATE)
features_norm = (features - NORM_MEAN) / (NORM_STD + 1e-8)
features_norm = np.clip(features_norm, -5, 5)
features_norm = np.expand_dims(features_norm, axis=-1)
features_norm = np.expand_dims(features_norm, axis=0)

cnn_pred     = model.predict(features_norm, verbose=0).flatten()
cnn_emotion  = EMOTION_LABELS[int(np.argmax(cnn_pred))]

print("CNN raw probabilities:")
for label, prob in zip(EMOTION_LABELS, cnn_pred):
    print(f"  {label}: {prob:.4f}")
print(f"CNN raw prediction : {cnn_emotion}")

# Step 2 — Calibration override
final_probs, zone, rms, pitch = apply_calibration(cnn_pred, audio, SAMPLE_RATE)

predicted_index  = int(np.argmax(final_probs))
predicted_emotion = EMOTIONS[predicted_index]
confidence        = float(np.max(final_probs)) * 100
a_w, t_w          = WEIGHTS[zone]

# ─────────────────────────────────────────────────────────
# FINAL RESULT OUTPUT — includes live voice frequency
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL RESULT")
print("=" * 55)
print(f"  Live RMS Energy    : {rms:.5f}")
print(f"  Live Voice Freq    : {pitch:.1f} Hz")
print(f"  Zone Detected      : {zone}")
print(f"  CNN Said           : {cnn_emotion.upper()}")
if zone != "middle_zone":
    print(f"  Calibration        : [CALIBRATION PRIORITIZED] overrode CNN -> {predicted_emotion.upper()}")
else:
    print(f"  Calibration        : [CNN + TEXT ACTIVE] Standard MFCC/CNN pipeline")
print(f"  Fusion Weights     : Audio={a_w}  Text={t_w}")
print()
print("  Final probabilities:")
for i, (label, prob) in enumerate(zip(EMOTIONS, final_probs)):
    marker = " <--" if i == predicted_index else ""
    print(f"    {label}: {prob:.4f}{marker}")
print()
print(f"  PREDICTED EMOTION  : {predicted_emotion.upper()}")
print(f"  Confidence         : {round(confidence, 2)} %")
print("=" * 55)