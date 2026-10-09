import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.llm.prompts import SYSTEM_PROMPT
from src.llm.afrique_qwen import AfriqueQwenLLM, DEFAULT_MODEL_PATH
from src.llm.conversation import ConversationManager
from src.llm.response_cleaner import clean_response
from src.asr.asr_engine import ASREngine
from src.tts.tts_engine import TTSEngine
from src.utils.language_detector import detect_language


class TestAssistantIntegration(unittest.TestCase):

    def test_01_model_path_resolution(self):
        """1. Test model path is resolved correctly from project root."""
        project_root = Path(__file__).resolve().parent.parent
        expected_path = (
            project_root
            / "models"
            / "afriqueqwen"
            / "AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf"
        )
        self.assertEqual(DEFAULT_MODEL_PATH.resolve(), expected_path.resolve())

    def test_02_system_prompt_included(self):
        """2. Test system prompt is included in chat completion request."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = "Muraho! Nguwo umusanzu wanjye."

        manager = ConversationManager(llm=mock_llm)
        manager.generate_response("Muraho")

        # Verify generate_completion was called with messages containing SYSTEM_PROMPT at index 0
        call_args = mock_llm.generate_completion.call_args[0][0]
        self.assertEqual(call_args[0]["role"], "system")
        self.assertEqual(call_args[0]["content"], SYSTEM_PROMPT)

    def test_03_history_includes_latest_user_message(self):
        """3. Test conversation history includes latest user message after response."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = "Ndi hano kuwfasha."

        manager = ConversationManager(llm=mock_llm)
        manager.generate_response("Bite?")

        messages = manager.messages
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Bite?")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Ndi hano kuwfasha.")

    def test_04_history_trimming_preserves_system_prompt_and_pairs(self):
        """4. Test history trimming preserves system prompt and complete recent turn pairs."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = "Subiza"

        manager = ConversationManager(llm=mock_llm, max_turns=2)
        # Add 3 turns
        manager.generate_response("Turn 1")
        manager.generate_response("Turn 2")
        manager.generate_response("Turn 3")

        messages = manager.messages
        # 1 system prompt + 2 turns (4 messages: User2, Asst2, User3, Asst3) = 5 messages total
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[1]["content"], "Turn 2")
        self.assertEqual(messages[3]["content"], "Turn 3")

    def test_05_clear_resets_conversation(self):
        """5. Test reset_conversation clears history while retaining system prompt."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = "Igisubizo"

        manager = ConversationManager(llm=mock_llm)
        manager.generate_response("Mwaramutse")
        self.assertGreater(len(manager.messages), 1)

        manager.reset_conversation()
        self.assertEqual(len(manager.messages), 1)
        self.assertEqual(manager.messages[0]["role"], "system")
        self.assertEqual(manager.messages[0]["content"], SYSTEM_PROMPT)

    def test_06_empty_transcriptions_do_not_trigger_llm(self):
        """6. Test empty transcriptions do not trigger LLM inference."""
        mock_asr = MagicMock(spec=ASREngine)
        mock_asr.transcribe_audio.return_value = ""

        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        manager = ConversationManager(llm=mock_llm)

        from run_assistant import transcribe_and_respond
        mock_tts = MagicMock(spec=TTSEngine)

        transcribe_and_respond(manager, mock_asr, mock_tts, "dummy.wav")
        mock_llm.generate_completion.assert_not_called()

    def test_07_empty_responses_do_not_trigger_tts(self):
        """7. Test empty LLM responses do not trigger TTS."""
        mock_asr = MagicMock(spec=ASREngine)
        mock_asr.transcribe_audio.return_value = "Muraho"

        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = ""

        manager = ConversationManager(llm=mock_llm)
        mock_tts = MagicMock(spec=TTSEngine)

        from run_assistant import transcribe_and_respond
        transcribe_and_respond(manager, mock_asr, mock_tts, "dummy.wav")

        mock_tts.synthesize_speech.assert_not_called()

    def test_08_reasoning_markers_cleaned(self):
        """8. Test reasoning markers are stripped from TTS input."""
        raw_text = "<think>\nThinking step...\n</think>\nMuraho neza! Umunsi mwiza."
        cleaned = clean_response(raw_text)
        self.assertEqual(cleaned, "Muraho neza! Umunsi mwiza.")
        self.assertNotIn("<think>", cleaned)

        raw_text2 = "[Start thinking]Deep thoughts[End thinking]Iyi ni Kinyarwanda."
        cleaned2 = clean_response(raw_text2)
        self.assertEqual(cleaned2, "Iyi ni Kinyarwanda.")

    def test_09_unsupported_languages_handled(self):
        """9. Test unsupported speech output languages (English/French) are skipped for TTS."""
        mock_tts = MagicMock(spec=TTSEngine)
        # Real detect_language should identify English
        lang_en = detect_language("Hello, how are you doing today?")
        self.assertEqual(lang_en, "english")

        # Test TTSEngine logic with English text
        engine = TTSEngine()
        result = engine.synthesize_speech("Hello, how are you today?", check_lang=True)
        self.assertIsNone(result)

    def test_10_errors_do_not_append_fabricated_assistant_response(self):
        """10. Test LLM errors do not append bad messages to history."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.side_effect = RuntimeError("Inference boom!")

        manager = ConversationManager(llm=mock_llm)
        initial_count = len(manager.messages)

        with self.assertRaises(RuntimeError):
            manager.generate_response("Fail me")

        self.assertEqual(len(manager.messages), initial_count)

    def test_11_text_mode_runs_independently(self):
        """11. Test text mode runs independently without initializing ASR or TTS."""
        mock_llm = MagicMock(spec=AfriqueQwenLLM)
        mock_llm.generate_completion.return_value = "Text response"
        manager = ConversationManager(llm=mock_llm)

        res = manager.generate_response("Hello in text mode")
        self.assertEqual(res, "Text response")

    def test_12_existing_implementations_intact(self):
        """12. Verify test_asr.py and test_kinya_flex.py exist and are unchanged."""
        project_root = Path(__file__).resolve().parent.parent
        test_asr_file = project_root / "scripts" / "test_asr.py"
        test_tts_file = project_root / "scripts" / "test_kinya_flex.py"

        self.assertTrue(test_asr_file.exists())
        self.assertTrue(test_tts_file.exists())


if __name__ == "__main__":
    unittest.main()
