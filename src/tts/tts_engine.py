import os
import wave
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch

from src.utils.language_detector import detect_language

# Resolve Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "kinya-flex-tts"
    / "kinya_flex_tts_base_trained.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tts_responses"


class TTSEngine:
    """
    Adapter for KinyaFlex TTS text-to-speech synthesis.
    Loads model once and synthesizes audio for Kinyarwanda text.
    """

    _instance = None

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        sample_rate: int = 24000,
        default_speaker: int = 0,
    ):
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.default_speaker = default_speaker
        self._model = None
        self._device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def load_model(self):
        """Loads FlexKinyaTTS model if not already loaded."""
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"KinyaFlex TTS model file not found at: {self.model_path.resolve()}"
            )

        try:
            from deepkin.models.flex_tts import FlexKinyaTTS
        except ImportError as e:
            raise ImportError(
                "deepkin package is required for KinyaFlex TTS. Ensure deepkin is installed."
            ) from e

        print(f"Loading KinyaFlex TTS model from: {self.model_path} (device={self._device})...")
        model = FlexKinyaTTS.from_pretrained(
            self._device,
            str(self.model_path.resolve()),
        )
        model.eval()
        self._model = model
        print("KinyaFlex TTS model loaded successfully!")
        return self._model

    def synthesize_speech(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speaker_id: Optional[int] = None,
        check_lang: bool = True,
    ) -> Optional[str]:
        """
        Synthesizes Kinyarwanda text into WAV audio file.
        Returns path to WAV file, or None if text is empty/unsupported language.
        """
        if not text or not text.strip():
            return None

        # Check language support
        if check_lang:
            lang = detect_language(text)
            if lang in ["english", "french"]:
                print(f"[TTS Notice] Text detected as {lang.capitalize()}. KinyaFlex TTS only supports Kinyarwanda. Skipping speech synthesis.")
                return None

        if speaker_id is None:
            speaker_id = self.default_speaker

        model = self.load_model()

        from deepkin.data.kinya_norm import text_to_sequence
        from deepkin.modules.tts_commons import intersperse

        text_id_sequence = intersperse(
            text_to_sequence(text, norm=True),
            0,
        )

        if len(text_id_sequence) == 0:
            print("[TTS Warning] Text converted to empty sequence.")
            return None

        with torch.no_grad():
            audio_data = model(text_id_sequence, speaker_id)

        # Process audio tensor to numpy PCM
        audio = audio_data.detach().cpu().numpy()
        audio = np.squeeze(audio)
        audio = np.clip(audio, -1.0, 1.0)
        pcm_audio = (audio * 32767).astype(np.int16)

        # Output file destination
        if output_path is None:
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_file = DEFAULT_OUTPUT_DIR / "latest_response.wav"
        else:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_file), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_audio.tobytes())

        print(f"[TTS Success] Synthesized audio saved to: {output_file.resolve()}")
        return str(output_file.resolve())
