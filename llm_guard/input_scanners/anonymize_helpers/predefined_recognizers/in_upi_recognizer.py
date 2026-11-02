from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class InUpiRecognizer(PatternRecognizer):
    """Recognizes UPI IDs near UPI/VPA context to avoid matching ordinary emails."""

    PATTERNS = [
        Pattern(
            "Indian UPI ID (Weak)",
            r"(?<![A-Za-z0-9._-])[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?@[A-Za-z][A-Za-z0-9.-]*[A-Za-z0-9](?![A-Za-z0-9.-])",
            0.4,
        ),
    ]

    CONTEXT = [
        "UPI",
        "VPA",
        "virtual payment address",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "IN_UPI",
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or self.PATTERNS,
            context=context or self.CONTEXT,
            supported_language=supported_language,
        )
