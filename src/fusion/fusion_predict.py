# src/fusion/fusion_predict.py
#
# Prediction pipeline: CNN + BiLSTM + Attention (audio) + DistilBERT (text)
# Output = weighted fusion of audio probs + text probs only.
#
# CALIBRATION (RMS / Pitch override) — COMMENTED OUT, not deleted.
# Zone-aware fusion weights replaced with fixed weights: audio=0.5, text=0.5
# (adjust below if you want to bias toward one modality).

import os
import numpy as np
import tensorflow as tf
import pickle
import whisper
import librosa
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from src.realtime.speech_to_text import record_audio
from src.audio.preprocess import extract_features

EXPECTED_TIME_FRAMES = 174

# ── Model paths ───────────────────────────────────────────────────────────────
AUDIO_MODEL_PATH   = "models/audio_cnn_bilstm_attention_best.keras"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
NORM_STATS_PATH    = "models/norm_stats.pkl"
TEXT_MODEL_DIR     = "models/text_emotion"

# Must match LabelEncoder alphabetical order: angry=0  fear=1  happy=2  sad=3
EMOTIONS = ["angry", "fear", "happy", "sad"]

# ── Fusion weights ────────────────────────────────────────────────────────────
# Pure audio + text, no calibration zones.
# Tweak these if one modality consistently outperforms the other for your voice.
AUDIO_WEIGHT = 0.5
TEXT_WEIGHT  = 0.5

# ────────────────────────────────────────────────────────────────────────────────
# CALIBRATION THRESHOLDS — commented out, NOT deleted.
# Uncomment and re-enable _calibration_override() + zone logic to restore.
# ────────────────────────────────────────────────────────────────────────────────
# ANGRY_RMS_THRESHOLD   = 0.09042
# ANGRY_PITCH_THRESHOLD = 209.4
# SAD_RMS_THRESHOLD     = 0.01881
# SAD_PITCH_THRESHOLD   = 150.0
#
# WEIGHTS = {
#     "angry_zone":  (0.65, 0.35),
#     "sad_zone":    (0.63, 0.37),
#     "middle_zone": (0.30, 0.70),
# }
# ────────────────────────────────────────────────────────────────────────────────

# ── Keyword override dictionary ───────────────────────────────────────────────
KEYWORD_OVERRIDE = {
    "happy": "happy", "happiness": "happy", "joy": "happy",
    "joyful": "happy", "excited": "happy", "cheerful": "happy",
    "glad": "happy", "delighted": "happy", "elated": "happy",

    "sad": "sad", "sadness": "sad", "unhappy": "sad",
    "depressed": "sad", "miserable": "sad", "gloomy": "sad",
    "sorrow": "sad", "sorrowful": "sad", "grief": "sad",

    "angry": "angry", "anger": "angry", "furious": "angry",
    "rage": "angry", "mad": "angry", "frustrated": "angry",
    "irritated": "angry", "annoyed": "angry",

    "fear": "fear", "fearful": "fear", "scared": "fear",
    "afraid": "fear", "terrified": "fear", "anxious": "fear",
    "nervous": "fear", "worried": "fear", "panic": "fear",
    "panicked": "fear",
}


def check_keyword_override(text):
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    for word in words:
        if word in KEYWORD_OVERRIDE:
            emotion = KEYWORD_OVERRIDE[word]
            print(f"  [Keyword Override] '{word}' → {emotion}")
            return emotion
    return None


# ── Load all models at import time ────────────────────────────────────────────
print("Loading models...")

audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
print(f"  Audio model loaded : {AUDIO_MODEL_PATH}")

with open(LABEL_ENCODER_PATH, "rb") as f:
    audio_le = pickle.load(f)

assert list(audio_le.classes_) == EMOTIONS, (
    f"EMOTIONS mismatch!\n  Expected: {EMOTIONS}\n  Got: {list(audio_le.classes_)}"
)

with open(NORM_STATS_PATH, "rb") as f:
    norm_stats = pickle.load(f)
NORM_MEAN = norm_stats["mean"]
NORM_STD  = norm_stats["std"]
print(f"  Norm stats → mean: {NORM_MEAN:.4f}  std: {NORM_STD:.4f}")

print("  Loading DistilBERT text model...")
tokenizer  = DistilBertTokenizer.from_pretrained(TEXT_MODEL_DIR)
text_model = DistilBertForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
text_model.eval()

whisper_model = whisper.load_model("small")
print("All models loaded.\n")


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe_text(file_path):
    result = whisper_model.transcribe(file_path, language="en")
    return result["text"].strip()


# ────────────────────────────────────────────────────────────────────────────────
# CALIBRATION FUNCTIONS — commented out, NOT deleted.
# ────────────────────────────────────────────────────────────────────────────────
# def _compute_acoustic_features(audio, sr):
#     """
#     Compute RMS energy and mean pitch (F0) from raw audio waveform.
#     Used by the calibration override layer only.
#     """
#     rms = float(np.sqrt(np.mean(audio ** 2)))
#     try:
#         f0, voiced_flag, _ = librosa.pyin(
#             audio,
#             fmin=librosa.note_to_hz('C2'),
#             fmax=librosa.note_to_hz('C7'),
#             sr=sr
#         )
#         voiced_f0  = f0[voiced_flag] if voiced_flag is not None else np.array([])
#         mean_pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
#     except Exception:
#         mean_pitch = 0.0
#     return rms, mean_pitch
#
#
# def _calibration_override(cnn_probs, audio, sr):
#     """
#     Post-processing calibration:
#       ANGRY : RMS > 0.09042 AND Pitch > 209.4 → override to angry (0.82)
#       SAD   : RMS < 0.01881 AND Pitch < 150.0 → override to sad   (0.82)
#       HAPPY/FEAR : no override
#     Returns: probs, zone, rms, pitch
#     """
#     probs = cnn_probs.copy()
#     rms, pitch = _compute_acoustic_features(audio, sr)
#
#     print(f"\n  [Calibration] RMS Energy : {rms:.5f}")
#     print(f"  [Calibration] Mean Pitch : {pitch:.1f} Hz")
#
#     zone = "middle_zone"
#     overridden = None
#
#     if rms > ANGRY_RMS_THRESHOLD and pitch > ANGRY_PITCH_THRESHOLD:
#         print(f"  [Calibration] ANGRY rule FIRED")
#         overridden = "angry"
#         zone = "angry_zone"
#     elif rms < SAD_RMS_THRESHOLD and pitch < SAD_PITCH_THRESHOLD:
#         print(f"  [Calibration] SAD rule FIRED")
#         overridden = "sad"
#         zone = "sad_zone"
#     else:
#         print("  [Calibration] No override — HAPPY/FEAR zone")
#
#     if overridden is not None:
#         idx        = EMOTIONS.index(overridden)
#         probs      = np.zeros(len(EMOTIONS))
#         probs[idx] = 0.82
#         other_prob = 0.18 / (len(EMOTIONS) - 1)
#         for i in range(len(EMOTIONS)):
#             if i != idx:
#                 probs[i] = other_prob
#         print(f"  [Calibration] Overridden probs: {np.round(probs, 4)}")
#
#     return probs, zone, rms, pitch
# ────────────────────────────────────────────────────────────────────────────────


# ── Audio prediction ──────────────────────────────────────────────────────────
def predict_audio(file_path):
    """
    Load audio → extract MFCC → CNN + BiLSTM + Attention → probabilities.
    No calibration applied. Returns raw model probabilities.
    """
    features = extract_features(file_path)
    if features is None:
        print("  [Audio] Feature extraction failed — returning zeros.")
        return np.zeros(len(EMOTIONS))

    features = (features - NORM_MEAN) / (NORM_STD + 1e-8)
    features = np.clip(features, -5, 5)
    features = features.reshape(1, 40, EXPECTED_TIME_FRAMES, 1).astype(np.float32)

    probs = audio_model.predict(features, verbose=0).flatten()
    probs = np.array(probs, dtype=float)
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)

    print(f"  [Audio CNN+BiLSTM+Attn]  "
          f"angry:{probs[0]:.3f}  fear:{probs[1]:.3f}  "
          f"happy:{probs[2]:.3f}  sad:{probs[3]:.3f}")

    # ──────────────────────────────────────────────────────────────────────────
    # CALIBRATION CALL — commented out, NOT deleted.
    # Replace the return below with these lines to restore calibration:
    # ──────────────────────────────────────────────────────────────────────────
    # audio, sr = librosa.load(file_path, sr=None)
    # calibrated_probs, zone, rms, pitch = _calibration_override(probs, audio, sr)
    # return calibrated_probs, zone, rms, pitch
    # ──────────────────────────────────────────────────────────────────────────

    return probs


# ── Text prediction ───────────────────────────────────────────────────────────
def predict_text(sentence):
    """
    Keyword override first, then DistilBERT if no keyword matched.
    Returns probability array of shape (4,).
    """
    overridden = check_keyword_override(sentence)
    if overridden and overridden in EMOTIONS:
        probs      = np.zeros(len(EMOTIONS))
        idx        = EMOTIONS.index(overridden)
        probs[idx] = 0.85
        other_prob = 0.15 / (len(EMOTIONS) - 1)
        for i in range(len(EMOTIONS)):
            if i != idx:
                probs[i] = other_prob
        print(f"  [Text] Keyword override → {overridden} (0.85 confidence)")
        return probs

    inputs = tokenizer(sentence, return_tensors="pt",
                       truncation=True, padding=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs   = F.softmax(outputs.logits, dim=1).cpu().numpy().flatten()

    mapped_probs = np.zeros(len(EMOTIONS))
    for i in range(min(len(probs), len(EMOTIONS))):
        mapped_probs[i] = probs[i]

    if np.sum(mapped_probs) > 0:
        mapped_probs = mapped_probs / np.sum(mapped_probs)

    print(f"  [Text DistilBERT]  "
          f"angry:{mapped_probs[0]:.3f}  fear:{mapped_probs[1]:.3f}  "
          f"happy:{mapped_probs[2]:.3f}  sad:{mapped_probs[3]:.3f}")

    return mapped_probs


# ── Fusion ────────────────────────────────────────────────────────────────────
def fuse_predictions(audio_probs, text_probs):
    """
    Fixed-weight fusion: audio * AUDIO_WEIGHT + text * TEXT_WEIGHT.
    No zone logic — calibration is commented out.

    To restore zone-aware fusion, uncomment WEIGHTS dict above and
    change signature to fuse_predictions(audio_probs, text_probs, zone).
    """
    audio_conf = float(np.max(audio_probs))
    text_conf  = float(np.max(text_probs))

    print(f"\n  Audio confidence : {audio_conf:.4f}")
    print(f"  Text  confidence : {text_conf:.4f}")
    print(f"  Fusion weights   : Audio={AUDIO_WEIGHT}  Text={TEXT_WEIGHT}")

    combined = audio_probs * AUDIO_WEIGHT + text_probs * TEXT_WEIGHT
    if np.sum(combined) > 0:
        combined = combined / np.sum(combined)

    print(f"  Fused probs      : {np.round(combined, 4)}")
    return EMOTIONS[int(np.argmax(combined))]

    # ──────────────────────────────────────────────────────────────────────────
    # ZONE-AWARE FUSION — commented out, NOT deleted.
    # ──────────────────────────────────────────────────────────────────────────
    # a_w, t_w = WEIGHTS[zone]
    # combined = audio_probs * a_w + text_probs * t_w
    # if np.sum(combined) > 0:
    #     combined = combined / np.sum(combined)
    # return EMOTIONS[int(np.argmax(combined))]
    # ──────────────────────────────────────────────────────────────────────────


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("\nRecording audio (7 seconds)...")
    audio_file = record_audio(duration=7)

    print("\nTranscribing...")
    text = transcribe_text(audio_file)
    print(f"Transcribed: {text}")

    print("\nAudio prediction (CNN + BiLSTM + Attention)...")
    audio_probs = predict_audio(audio_file)
    print(f"Audio emotion: {EMOTIONS[np.argmax(audio_probs)].upper()}")

    print("\nText prediction (DistilBERT)...")
    text_probs = predict_text(text)
    print(f"Text emotion: {EMOTIONS[np.argmax(text_probs)].upper()}")

    fmt = "  angry:{:.3f}  fear:{:.3f}  happy:{:.3f}  sad:{:.3f}"
    print(f"\nAudio probs : {fmt.format(*audio_probs)}")
    print(f"Text  probs : {fmt.format(*text_probs)}")

    print("\nFusing...")
    final_emotion = fuse_predictions(audio_probs, text_probs)

    print("\n" + "=" * 50)
    print("  FINAL RESULT")
    print("=" * 50)
    print(f"  Transcribed Text  : {text}")
    print(f"  Audio Emotion     : {EMOTIONS[np.argmax(audio_probs)].upper()}")
    print(f"  Text  Emotion     : {EMOTIONS[np.argmax(text_probs)].upper()}")
    print(f"  Fusion Weights    : Audio={AUDIO_WEIGHT}  Text={TEXT_WEIGHT}")
    print(f"\n  PREDICTED EMOTION : {final_emotion.upper()}")
    print("=" * 50)


# ── Pipeline function (called from frontend / API) ────────────────────────────
def run_fusion_pipeline():
    audio_file = record_audio(duration=10)
    if audio_file is None:
        raise ValueError("Audio recording failed")

    text        = transcribe_text(audio_file)
    audio_probs = predict_audio(audio_file)
    text_probs  = predict_text(text)
    final       = fuse_predictions(audio_probs, text_probs)

    return {
        "text":          text,
        "audio_emotion": EMOTIONS[np.argmax(audio_probs)],
        "text_emotion":  EMOTIONS[np.argmax(text_probs)],
        "final_emotion": final,
        "audio_probs":   audio_probs.tolist(),
        "text_probs":    text_probs.tolist(),
    }

    # ──────────────────────────────────────────────────────────────────────────
    # CALIBRATION VERSION of run_fusion_pipeline — commented out, NOT deleted.
    # ──────────────────────────────────────────────────────────────────────────
    # text                          = transcribe_text(audio_file)
    # audio_probs, zone, rms, pitch = predict_audio(audio_file)   # calibrated
    # text_probs                    = predict_text(text)
    # final                         = fuse_predictions(audio_probs, text_probs, zone)
    # return {
    #     "text":          text,
    #     "audio_emotion": EMOTIONS[np.argmax(audio_probs)],
    #     "text_emotion":  EMOTIONS[np.argmax(text_probs)],
    #     "final_emotion": final,
    #     "rms_energy":    rms,
    #     "voice_freq_hz": pitch,
    #     "zone":          zone,
    # }
    # ──────────────────────────────────────────────────────────────────────────