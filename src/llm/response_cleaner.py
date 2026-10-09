import re


def clean_response(text: str) -> str:
    """
    Cleans the model's generated text response for display and TTS playback.

    Removes internal reasoning markers, control tokens, and unwanted formatting
    while preserving punctuation, Kinyarwanda/French characters, and meaning.
    """
    if not text:
        return ""

    cleaned = text

    # Remove standard control tokens if emitted
    control_tokens = [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
    ]
    for token in control_tokens:
        cleaned = cleaned.replace(token, "")

    # Remove explicit thinking blocks if present e.g. <think>...</think>
    # Handle complete <think>...</think> blocks
    if "<think>" in cleaned and "</think>" in cleaned:
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    elif "<think>" in cleaned and "</think>" not in cleaned:
        # If <think> tag is unclosed, check if there's text after it or if the whole thing is thinking
        parts = cleaned.split("<think>", 1)
        cleaned = parts[0]

    # Handle [Start thinking]...[End thinking] blocks
    if "[Start thinking]" in cleaned and "[End thinking]" in cleaned:
        cleaned = re.sub(r"\[Start thinking\].*?\[End thinking\]", "", cleaned, flags=re.DOTALL)
    elif "[Start thinking]" in cleaned and "[End thinking]" not in cleaned:
        parts = cleaned.split("[Start thinking]", 1)
        cleaned = parts[0]

    # Remove standalone residual tags
    residual_tags = ["<think>", "</think>", "[Start thinking]", "[End thinking]"]
    for tag in residual_tags:
        cleaned = cleaned.replace(tag, "")

    # Clean leading/trailing whitespaces and extra internal line breaks
    cleaned = cleaned.strip()

    return cleaned
