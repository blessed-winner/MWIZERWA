from pathlib import Path

import torch
from transformers import pipeline


MODEL_ID = "ElvisTata2024/Kinyarwanda-Health-ASR"

device = 0 if torch.cuda.is_available() else -1

print("Loading ASR model...")

asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=device,
)

print("ASR model loaded!")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
audio_file = PROJECT_ROOT / "test.wav"

print(f"Audio file: {audio_file}")

if not audio_file.exists():
    raise FileNotFoundError(f"Audio file not found: {audio_file}")

result = asr(str(audio_file))

print("\nTranscription:")
print(result["text"])