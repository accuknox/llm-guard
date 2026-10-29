from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class InBankAccountRecognizer(PatternRecognizer):
    """Recognizes numeric Indian bank account numbers near account-related context."""

    PATTERNS = [
        Pattern(
            "Indian Bank Account (Weak)",
            r"(?<!\d)\d{9,18}(?!\d)",
            0.4,
        ),
    ]

    CONTEXT = [
        "account",
        "bank",
        "account no",
        "account number",
        "a/c",
        "a/c no",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "IN_BANK_ACCOUNT",
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or self.PATTERNS,
            context=context or self.CONTEXT,
            supported_language=supported_language,
        )
