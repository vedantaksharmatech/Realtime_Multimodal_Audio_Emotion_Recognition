# src/audio/predict.py
#
# Single-file emotion prediction using CNN + BiLSTM + Attention model.
# Output: audio prediction only.
# Calibration / text-fusion output COMMENTED OUT — not deleted.

import numpy as np
import tensorflow as tf
import pickle
from src.audio.preprocess import extract_features

# Best model checkpoint path
MODEL_PATH = "models/audio_cnn_bilstm_attention_best.keras"


def predict_emotion(file_path):
    """
    Predict emotion from a .wav file using CNN + BiLSTM + Attention.
    Returns (emotion_str, probabilities_array) or (None, None) on error.
    """

    model = tf.keras.models.load_model(MODEL_PATH)

    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("models/norm_stats.pkl", "rb") as f:
        norm_stats = pickle.load(f)
    norm_mean = norm_stats["mean"]
    norm_std  = norm_stats["std"]

    # Extract MFCC
    features = extract_features(file_path)
    if features is None:
        print("Feature extraction failed.")
        return None, None

    # Normalise → (1, 40, 174, 1)
    features = (features - norm_mean) / (norm_std + 1e-8)
    features = np.clip(features, -5, 5)
    features = features[..., np.newaxis]
    features = np.expand_dims(features, axis=0)

    # Predict
    prediction = model.predict(features, verbose=0)
    pred_idx   = int(np.argmax(prediction, axis=1)[0])
    emotion    = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(prediction[0][pred_idx])

    # ── Audio-only output ─────────────────────────────────────────────────────
    print("=" * 45)
    print("  AUDIO PREDICTION  [CNN + BiLSTM + Attention]")
    print("=" * 45)
    print(f"  Predicted Emotion : {emotion.upper()}")
    print(f"  Confidence        : {confidence:.2%}")
    print()
    print("  All class probabilities:")
    for label, prob in zip(label_encoder.classes_, prediction[0]):
        bar = "█" * int(prob * 20)
        print(f"    {label:<8} {prob:.4f}  {bar}")
    print("=" * 45)

    # ─────────────────────────────────────────────────────────────────────────
    # CALIBRATION OUTPUT — commented out, NOT deleted.
    # ─────────────────────────────────────────────────────────────────────────
    # import os
    # cal_path = "models/audio_cnn_bilstm_attention_calibrated.keras"
    # if os.path.exists(cal_path):
    #     cal_model   = tf.keras.models.load_model(cal_path)
    #     cal_pred    = cal_model.predict(features, verbose=0)
    #     cal_idx     = int(np.argmax(cal_pred, axis=1)[0])
    #     cal_emotion = label_encoder.inverse_transform([cal_idx])[0]
    #     cal_conf    = float(cal_pred[0][cal_idx])
    #     print(f"\n  Calibrated Emotion    : {cal_emotion.upper()}")
    #     print(f"  Calibrated Confidence : {cal_conf:.2%}")
    #     for label, prob in zip(label_encoder.classes_, cal_pred[0]):
    #         print(f"    {label:<8} {prob:.4f}")
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT + AUDIO FUSION OUTPUT — commented out, NOT deleted.
    # ─────────────────────────────────────────────────────────────────────────
    # from src.text.predict import predict_text_emotion
    # text_probs  = predict_text_emotion(transcript)
    # audio_probs = prediction[0]
    # AUDIO_WEIGHT = 0.6
    # TEXT_WEIGHT  = 0.4
    # fused_probs  = AUDIO_WEIGHT * audio_probs + TEXT_WEIGHT * text_probs
    # fused_idx    = int(np.argmax(fused_probs))
    # fused_emotion = label_encoder.inverse_transform([fused_idx])[0]
    # print(f"\n  FUSED Prediction  : {fused_emotion.upper()}")
    # for label, prob in zip(label_encoder.classes_, fused_probs):
    #     print(f"    {label:<8} {prob:.4f}")
    # ─────────────────────────────────────────────────────────────────────────

    return emotion, prediction[0]


if __name__ == "__main__":
    import os
    test_file = "data/CREMAD/1001_DFA_ANG_XX.wav"   # update to a valid CREMA-D file
    if os.path.exists(test_file):
        predict_emotion(test_file)
    else:
        print(f"Test file not found: {test_file}")
        print("Update 'test_file' to any valid CREMA-D .wav path.")