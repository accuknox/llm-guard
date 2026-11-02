from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class InIfscRecognizer(PatternRecognizer):
    """Recognizes the standard structure of Indian IFSC codes."""

    PATTERNS = [
        Pattern(
            "Indian IFSC Code",
            r"(?i)\b[A-Z]{4}0[A-Z0-9]{6}\b",
            0.75,
        ),
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        supported_language: str = "en",
        supported_entity: str = "IN_IFSC",
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or self.PATTERNS,
            supported_language=supported_language,
        )
