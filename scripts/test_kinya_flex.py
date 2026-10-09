import os
import wave

import numpy as np
import torch

from deepkin.data.kinya_norm import text_to_sequence
from deepkin.models.flex_tts import FlexKinyaTTS
from deepkin.modules.tts_commons import intersperse


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    "models/kinya-flex-tts/"
    "kinya_flex_tts_base_trained.pt"
)

OUTPUT_DIR = "outputs/kinya-flex-final-test"

TEXT = (
    "Mu buzima bwa buri munsi, abantu bahura n’ibibazo bitandukanye "
    "bisaba gutekereza, kwihangana no gufata ibyemezo bifite ishingiro. "
    "Ikoranabuhanga rishobora kugira uruhare runini mu koroshya imirimo "
    "no gutuma abantu babona amakuru bakeneye mu gihe gito, ariko "
    "rigomba gukoreshwa mu buryo bwubahiriza umuntu n’imibereho ye. "
    "Mu Rwanda, gukoresha ikoranabuhanga mu rurimi rw’Ikinyarwanda "
    "bishobora gufasha abantu benshi kubona serivisi n’amakuru mu buryo "
    "buboroheye. Ni yo mpamvu ari ngombwa guteza imbere sisitemu zumva "
    "kandi zivuga Ikinyarwanda neza, ku buryo ijwi ryazo ryumvikana, "
    "amagambo akavugwa neza, kandi imvugo ikagira injyana isanzwe "
    "nk’iy’umuntu."
)

SAMPLE_RATE = 24000

# Test all three voices
SPEAKERS = {
    0: "female_1",
    1: "female_2",
    2: "male",
}


# ============================================================
# Setup
# ============================================================

device = (
    torch.device("cuda:0")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("=" * 60)
print("Kinya-Flex TTS Final Qualitative Test")
print("=" * 60)

print(f"Device: {device}")
print(f"Model:  {MODEL_PATH}")
print()

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load model
# ============================================================

print("Loading model...")

kinya_tts = FlexKinyaTTS.from_pretrained(
    device,
    MODEL_PATH
)

kinya_tts.eval()

print("Model loaded successfully.")
print()


# ============================================================
# Process text
# ============================================================

print("Processing Kinyarwanda text...")

text_id_sequence = intersperse(
    text_to_sequence(TEXT, norm=True),
    0
)

print(f"Text length: {len(TEXT)} characters")
print(f"Token length: {len(text_id_sequence)}")
print()


# ============================================================
# Generate audio for each speaker
# ============================================================

for speaker_id, speaker_name in SPEAKERS.items():

    print("-" * 60)
    print(f"Generating Speaker {speaker_id} ({speaker_name})...")
    print("-" * 60)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"speaker_{speaker_id}_{speaker_name}.wav"
    )

    # Generate speech
    with torch.no_grad():
        audio_data = kinya_tts(
            text_id_sequence,
            speaker_id
        )

    print("Audio generated.")

    # --------------------------------------------------------
    # Convert tensor -> NumPy
    # --------------------------------------------------------

    audio = audio_data.detach().cpu().numpy()

    # Remove dimensions such as [1, samples] -> [samples]
    audio = np.squeeze(audio)

    # Make sure values are within [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)

    # Convert float audio to 16-bit PCM
    audio = (audio * 32767).astype(np.int16)

    # --------------------------------------------------------
    # Save WAV
    # --------------------------------------------------------

    print("Saving WAV...")

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit PCM
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())

    duration = len(audio) / SAMPLE_RATE

    print(f"Saved:    {output_path}")
    print(f"Duration: {duration:.2f} seconds")
    print()


# ============================================================
# Summary
# ============================================================

print("=" * 60)
print("FINAL TEST COMPLETE")
print("=" * 60)

print()
print("Text used:")
print(TEXT)
print()

print("Generated files:")

for speaker_id, speaker_name in SPEAKERS.items():
    path = os.path.join(
        OUTPUT_DIR,
        f"speaker_{speaker_id}_{speaker_name}.wav"
    )

    print(f"  Speaker {speaker_id} ({speaker_name}): {path}")

print()
print("Listen to all three recordings and compare:")
print("  1. Pronunciation")
print("  2. Intelligibility")
print("  3. Naturalness")
print("  4. Rhythm and pauses")
print("  5. Consistency throughout the paragraph")
print()
print("Test complete.")