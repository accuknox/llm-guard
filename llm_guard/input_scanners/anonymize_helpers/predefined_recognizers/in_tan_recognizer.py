from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class InTanRecognizer(PatternRecognizer):
    """Recognizes the 10-character structure of Indian TANs."""

    PATTERNS = [
        Pattern(
            "Indian TAN",
            r"(?i)\b[A-Z]{4}\d{5}[A-Z]\b",
            0.75,
        ),
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        supported_language: str = "en",
        supported_entity: str = "IN_TAN",
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or self.PATTERNS,
            supported_language=supported_language,
        )
