import os
from pathlib import Path
from typing import List, Dict, Any

# Resolve Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "afriqueqwen"
    / "AfriqueQwen3.5-4B-Instruct-v1.i1-Q4_K_M.gguf"
)


class AfriqueQwenLLM:
    """
    Wrapper for local AfriqueQwen3.5 GGUF model using llama-cpp-python.
    Ensures model is loaded once and reused across inference requests.
    """

    _instance = None

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._llm = None

    def load_model(self):
        """Loads the GGUF model if not already loaded."""
        if self._llm is not None:
            return self._llm

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"AfriqueQwen model GGUF file not found at: {self.model_path.resolve()}"
            )

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for AfriqueQwen inference. "
                "Please install llama-cpp-python."
            ) from e

        print(f"Loading AfriqueQwen model from: {self.model_path} ...")
        self._llm = Llama(
            model_path=str(self.model_path.resolve()),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        print("AfriqueQwen model loaded successfully!")
        return self._llm

    def generate_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Generates a completion from a list of chat messages using the model's chat template.
        """
        llm = self.load_model()
        response = llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        choices = response.get("choices", [])
        if not choices:
            return ""

        content = choices[0].get("message", {}).get("content", "")
        return content or ""
