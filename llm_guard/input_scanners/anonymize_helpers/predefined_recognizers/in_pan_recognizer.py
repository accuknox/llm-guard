from presidio_analyzer import Pattern
from presidio_analyzer.predefined_recognizers import InPanRecognizer as PresidioInPanRecognizer


class InPanRecognizer(PresidioInPanRecognizer):
    """Recognizes Indian PAN numbers written with spaces or dashes.

    A PAN is five letters, four digits and a letter (``ABCPE1234F``), where the fourth
    letter is the holder type. Presidio only matches the compact form; this also accepts
    a single space or dash between the groups (``ABCPE 1234 F``, ``ABCPE-1234-F``,
    ``ABCPE 1234F``). The separated forms must be uppercase, since Presidio matches
    case-insensitively and ordinary text like "world 2024 a" would otherwise match.
    """

    PATTERNS = [
        Pattern(
            "PAN (High)",
            r"\b[A-Za-z]{3}[AaBbCcFfGgHhJjLlPpTt][A-Za-z]\d{4}[A-Za-z]\b",
            0.85,
        ),
        Pattern(
            "PAN (High, separated)",
            r"\b(?-i:[A-Z]{3}[ABCFGHJLPT][A-Z][- ]?\d{4}[- ]?[A-Z])\b",
            0.85,
        ),
        Pattern(
            "PAN (Medium)",
            r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b",
            0.6,
        ),
        Pattern(
            "PAN (Medium, separated)",
            r"\b(?-i:[A-Z]{5}[- ]?\d{4}[- ]?[A-Z])\b",
            0.6,
        ),
    ]
