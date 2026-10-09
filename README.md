# MINDORA - Kinyarwanda Local Voice Assistant

An integrated, fully local voice assistant powered by:
1. **ASR (Speech Recognition):** `ElvisTata2024/Kinyarwanda-Health-ASR` (HuggingFace Transformers pipeline).
2. **LLM (Language Model):** `AfriqueQwen3.5-4B-Instruct` GGUF loaded locally via `llama-cpp-python`.
3. **TTS (Text-to-Speech):** `KinyaFlex TTS` (`FlexKinyaTTS` PyTorch model).

---

## 1. Architecture Overview

```
[ Audio WAV File / Microphone ]
               │
               ▼
┌───────────────────────────────┐
│          ASR Engine           │ (Kinyarwanda Health ASR)
└──────────────┬────────────────┘
               │ Transcribed Text
               ▼
┌───────────────────────────────┐
│     Conversation Manager      │ (History Trimming & System Prompt)
└──────────────┬────────────────┘
               │ Prompt + Context
               ▼
┌───────────────────────────────┐
│       AfriqueQwen LLM         │ (AfriqueQwen3.5-4B-Instruct GGUF)
└──────────────┬────────────────┘
               │ Raw LLM Output
               ▼
┌───────────────────────────────┐
│       Response Cleaner        │ (Strips <think> tags & control tokens)
└──────────────┬────────────────┘
               │ Cleaned Text
               ▼
┌───────────────────────────────┐
│       Language Detector       │ (Ensures Kinyarwanda before TTS)
└──────────────┬────────────────┘
               │ Kinyarwanda Text
               ▼
┌───────────────────────────────┐
│        KinyaFlex TTS          │ (Synthesizes 24kHz 16-bit PCM WAV)
└───────────────────────────────┘
```

---

## 2. Project Directory & Model Paths

Local model weights are excluded from Git repository tracking via `.gitignore`. Before running, place model files in the following paths:

- **AfriqueQwen LLM GGUF:**
  `models/afriqueqwen/AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf`
- **KinyaFlex TTS Checkpoint:**
  `models/kinya-flex-tts/kinya_flex_tts_base_trained.pt`
- **ASR Model:**
  `ElvisTata2024/Kinyarwanda-Health-ASR` (HuggingFace repository cached locally on first run)

---

## 3. Environment Setup & Installation

Ensure Python 3.12 is active and install required dependencies:

```powershell
# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## 4. How to Run

### Text Mode (Console Chat)

Run text mode independently without loading audio models:

```bash
python run_assistant.py --mode text
```

**Commands in Text Mode:**
- `/clear` – Resets conversation history while preserving the `SYSTEM_PROMPT`.
- `exit` or `quit` – Exits the application.

### Voice Mode (End-to-End Pipeline)

Run full voice mode pipeline:

```bash
python run_assistant.py --mode voice
```

Or pass a specific WAV audio file for single-turn testing:

```bash
python run_assistant.py --mode voice --audio-file test.wav
```

Speaker voices for KinyaFlex TTS can be specified with `--speaker`:
- `0`: female_1 (default)
- `1`: female_2
- `2`: male

Example:
```bash
python run_assistant.py --mode voice --audio-file test.wav --speaker 1
```

---

## 5. Independent Model Testing

### Testing ASR Independently

```bash
python scripts/test_asr.py
```
Or via engine:
```bash
python -c "from src.asr.asr_engine import ASREngine; print(ASREngine().transcribe_audio('test.wav'))"
```

### Testing TTS Independently

```bash
python scripts/test_kinya_flex.py
```
Or via engine:
```bash
python -c "from src.tts.tts_engine import TTSEngine; TTSEngine().synthesize_speech('Muraho neza, amakuru yanyu?')"
```

### Running Unit Tests

Run the test suite (12 unit tests using mocks):

```bash
python -m unittest tests/test_assistant.py
```

---

## 6. Supported Languages & Behavior

- **ASR Input:** Supports Kinyarwanda speech recognition.
- **LLM Output:** Can converse in **Kinyarwanda**, **English**, and **French**.
- **TTS Output:** **Kinyarwanda only.** If the assistant responds in English or French, the response text is printed to the console, and speech synthesis is skipped with an informative notice.
- **Therapeutic Guidance:** The system prompt instructs the assistant to listen, clarify, and understand before offering low-risk suggestions. It does not spam generic exercises when user mentions ordinary difficulties.

---

## 7. Current Limitations & Troubleshooting

1. **CPU Inference:** High-resource CPU setup. LLM generation may take several seconds per turn.
2. **Audio Hardware:** If recording via microphone fails on Windows, use `--audio-file <path.wav>` to pass audio recordings.
3. **`llama-cpp-python` Build Issues:** If building `llama-cpp-python` from source fails due to system header conflicts (e.g. MSYS in `PATH`), remove MSYS directories from `PATH` before running `pip install llama-cpp-python`.
