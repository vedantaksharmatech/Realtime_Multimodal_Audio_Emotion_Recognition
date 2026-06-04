# import torch
# import torch.nn.functional as F
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# from transformers import (
#     DistilBertTokenizerFast,
#     DistilBertForSequenceClassification,
#     WhisperProcessor,
#     WhisperForConditionalGeneration
# )

# # =========================
# # LOAD TEXT EMOTION MODEL
# # =========================

# model_path = "models/text_emotion"

# tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
# text_model = DistilBertForSequenceClassification.from_pretrained(model_path)
# text_model.eval()

# # =========================
# # LOAD WHISPER MODEL
# # =========================

# print("Loading Whisper speech-to-text model...")
# whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
# whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
# whisper_model.eval()
# print("Models loaded successfully.\n")

# # =========================
# # GOEMOTIONS LABELS
# # =========================

# labels = [
#     "admiration","amusement","anger","annoyance","approval","caring",
#     "confusion","curiosity","desire","disappointment","disapproval",
#     "disgust","embarrassment","excitement","fear","gratitude","grief",
#     "joy","love","nervousness","optimism","pride","realization",
#     "relief","remorse","sadness","surprise","neutral"
# ]

# emotion_groups = {
#     "happy": [
#         "admiration","amusement","approval","caring","desire",
#         "excitement","gratitude","joy","love","optimism",
#         "pride","relief","surprise"
#     ],
#     "sad": [
#         "sadness","disappointment","grief","remorse","embarrassment"
#     ],
#     "angry": [
#         "anger","annoyance","disapproval","disgust"
#     ],
#     "fear": [
#         "fear","nervousness","confusion",
#         "curiosity","realization","neutral"
#     ]
# }

# # =========================
# # RECORD AUDIO
# # =========================

# def record_audio(duration=5, fs=16000):
#     print("Recording... Speak now.")
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#     sd.wait()
#     print("Recording complete.\n")
#     return audio.flatten(), fs


# # =========================
# # TRANSCRIBE AUDIO
# # =========================

# def transcribe_audio(audio, fs):
#     inputs = whisper_processor(audio, sampling_rate=fs, return_tensors="pt")
#     with torch.no_grad():
#         predicted_ids = whisper_model.generate(inputs.input_features)
#     transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#     return transcription


# # =========================
# # TEXT EMOTION PREDICTION
# # =========================

# def predict_emotion(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True)

#     with torch.no_grad():
#         outputs = text_model(**inputs)

#     probs = F.softmax(outputs.logits, dim=1)[0]

#     label_probs = {
#         labels[i]: probs[i].item()
#         for i in range(len(labels))
#     }

#     grouped_scores = {}
#     for group, group_labels in emotion_groups.items():
#         grouped_scores[group] = sum(label_probs[l] for l in group_labels)

#     final_emotion = max(grouped_scores, key=grouped_scores.get)

#     return final_emotion


# # =========================
# # MAIN LOOP
# # =========================

# while True:
#     user_input = input("Press Enter to speak or type 'exit': ")

#     if user_input.lower() == "exit":
#         break

#     audio, fs = record_audio(duration=5)

#     text = transcribe_audio(audio, fs)
#     print("Transcribed Text:", text)

#     emotion = predict_emotion(text)
#     print("Predicted Emotion:", emotion)
#     print("-" * 40)

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="models/text_emotion",
    tokenizer="models/text_emotion"
)

print("\nHappy:")
print(classifier("I am extremely happy today"))

print("\nFear:")
print(classifier("I am scared"))

print("\nAngry:")
print(classifier("I am very angry"))

print("\nSad:")
print(classifier("I feel sad"))