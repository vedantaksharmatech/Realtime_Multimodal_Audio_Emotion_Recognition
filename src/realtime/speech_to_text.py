import whisper
import sounddevice as sd
import numpy as np
import wave
import os

# Load Whisper model
model = whisper.load_model("small")  # or "tiny" for faster

def record_audio(duration=5, fs=16000):
    """
    Record audio from the microphone for the specified duration.
    Saves the recorded audio as a permanent WAV file in src/realtime folder.
    """
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    recording = np.squeeze(recording)

    # Define permanent file path
    folder_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, "recorded_audio.wav")

    # Save recording to permanent WAV file
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())
    
    print(f"Saved recording as {filename}")
    return filename  # return the permanent file path

def transcribe_audio(file_path):
    """
    Transcribe the given audio file using Whisper model.
    """
    result = model.transcribe(file_path)
    return result["text"]

if __name__ == "__main__":
    # Record 5 seconds of audio
    audio_file = record_audio(duration=5)
    # Transcribe the recorded audio
    text = transcribe_audio(audio_file)
    print("Transcribed text:", text)