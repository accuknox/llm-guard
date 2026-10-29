from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class InDrivingLicenseRecognizer(PatternRecognizer):
    """Recognizes common Indian driving licence number formats."""

    PATTERNS = [
        Pattern(
            "Indian Driving Licence (Weak)",
            r"(?<![A-Z0-9])[A-Z]{2}[-/ ]?\d{2}[-/ ]?\d{4}[-/ ]?\d{5,7}(?![A-Z0-9])",
            0.35,
        ),
    ]

    CONTEXT = [
        "driving",
        "licence",
        "license",
        "DL",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "IN_DRIVING_LICENSE",
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or self.PATTERNS,
            context=context or self.CONTEXT,
            supported_language=supported_language,
        )
