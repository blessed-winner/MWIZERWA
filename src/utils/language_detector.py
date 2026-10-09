import re


def detect_language(text: str) -> str:
    """
    Simple heuristic language detector for Kinyarwanda, English, and French.
    Returns: 'kinyarwanda', 'english', 'french', or 'unknown'.
    """
    if not text or not text.strip():
        return "unknown"

    words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    if not words:
        return "unknown"

    words_set = set(words)

    # Marker words for English
    en_markers = {
        "the", "is", "are", "you", "and", "to", "in", "that", "it", "for", "this",
        "with", "on", "have", "hello", "thank", "thanks", "yes", "no", "what",
        "how", "why", "can", "help", "good", "morning", "feel", "day", "trouble"
    }

    # Marker words for French
    fr_markers = {
        "le", "la", "les", "des", "du", "un", "une", "est", "sont", "vous", "nous",
        "avec", "pour", "dans", "cette", "ce", "que", "qui", "mais", "oui", "non",
        "bonjour", "merci", "comment", "pourquoi", "bien", "journée", "problème"
    }

    # Marker words for Kinyarwanda
    rw_markers = {
        "ni", "ko", "mu", "ku", "na", "ba", "za", "ya", "wa", "za", "ariko", "kandi",
        "nko", "ngo", "nka", "bantu", "umuntu", "muraho", "urakoze", "bite", "yego",
        "oya", "ubuzima", "rwanda", "kinyarwanda", "mwiriwe", "mwaramutse", "mbona",
        "ngaho", "amatsiko", "ikibazo", "shaka", "gushaka", "kora", "gukora", "bivuga",
        "bifite", "bwa", "bya", "rya", "muri"
    }

    en_score = len(words_set.intersection(en_markers))
    fr_score = len(words_set.intersection(fr_markers))
    rw_score = len(words_set.intersection(rw_markers))

    # Kinyarwanda characteristic patterns (e.g. apostrophes in words like n'ibibazo, rw'ikinyarwanda, kw'ishuri)
    if re.search(r"\b\w+’\w+\b|\b\w+'\w+\b", text):
        rw_score += 2

    # Check max score
    if rw_score > en_score and rw_score > fr_score:
        return "kinyarwanda"
    elif en_score > rw_score and en_score >= fr_score:
        return "english"
    elif fr_score > rw_score and fr_score > en_score:
        return "french"

    # Default to kinyarwanda if ambiguous (since AfriqueQwen prompt & task focus on Kinyarwanda assistant)
    return "kinyarwanda"
