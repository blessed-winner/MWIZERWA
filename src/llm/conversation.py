from typing import List, Dict, Optional
from src.llm.prompts import SYSTEM_PROMPT
from src.llm.afrique_qwen import AfriqueQwenLLM
from src.llm.response_cleaner import clean_response


class ConversationManager:
    """
    Manages conversational history with AfriqueQwen, ensuring:
    1. System prompt is always preserved at index 0.
    2. Context window is bounded by keeping max recent turn pairs.
    3. User/assistant messages are only appended upon successful inference.
    4. Model responses are cleaned of thinking/control tokens before return.
    """

    def __init__(
        self,
        llm: Optional[AfriqueQwenLLM] = None,
        max_turns: int = 6,
    ):
        self.llm = llm if llm is not None else AfriqueQwenLLM()
        self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []
        self.reset_conversation()

    def reset_conversation(self) -> None:
        """Resets the conversation history, preserving the system prompt."""
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

    def _trim_history(self) -> None:
        """
        Trims older user-assistant turn pairs when history exceeds max_turns.
        Always retains system prompt at index 0.
        """
        # Exclude system prompt (index 0)
        context = self.messages[1:]
        max_messages = self.max_turns * 2  # Each turn is (user, assistant)

        while len(context) > max_messages:
            # Remove the oldest turn pair (2 messages)
            context = context[2:]

        self.messages = [self.messages[0]] + context

    def generate_response(self, user_text: str) -> str:
        """
        Generates a response for a user message, maintaining context and handling errors.
        """
        user_text = user_text.strip() if user_text else ""
        if not user_text:
            return ""

        # Prepare payload with current user message appended
        candidate_messages = self.messages + [
            {
                "role": "user",
                "content": user_text,
            }
        ]

        try:
            # Inference call
            raw_response = self.llm.generate_completion(candidate_messages)
            cleaned = clean_response(raw_response)

            if not cleaned:
                # Empty response generated
                return ""

            # Append successful user and assistant turn to official state
            self.messages.append({"role": "user", "content": user_text})
            self.messages.append({"role": "assistant", "content": cleaned})

            # Trim history if needed
            self._trim_history()

            return cleaned
        except Exception as e:
            # On error, do not mutate state with half-turns
            print(f"Error during LLM inference: {e}")
            raise e
