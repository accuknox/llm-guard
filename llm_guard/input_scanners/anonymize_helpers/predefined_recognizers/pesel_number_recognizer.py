from datetime import date

from presidio_analyzer import Pattern, PatternRecognizer


class PeselNumberRecognizer(PatternRecognizer):
    """Recognizes Polish PESEL personal identification numbers."""

    PATTERNS = [
        Pattern(
            "PESEL Number (High)",
            r"\b\d{11}\b",
            0.6,
        ),
    ]

    CONTEXT = [
        "pesel",
        "polish id",
        "personal identification number",
        "polish identification number",
    ]

    def __init__(
        self,
        patterns: list[Pattern] | None = None,
        context: list[str] | None = None,
        supported_language: str = "en",
        supported_entity: str = "PESEL_NUMBER",
    ):
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or self.PATTERNS,
            context=context or self.CONTEXT,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        if len(pattern_text) != 11 or not pattern_text.isdigit():
            return False

        month_code = int(pattern_text[2:4])
        month = month_code % 20
        year_base = {
            0: 1900,
            20: 2000,
            40: 2100,
            60: 2200,
            80: 1800,
        }.get(month_code - month)
        if year_base is None or not 1 <= month <= 12:
            return False

        try:
            date(year_base + int(pattern_text[:2]), month, int(pattern_text[4:6]))
        except ValueError:
            return False

        weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
        checksum = sum(int(digit) * weight for digit, weight in zip(pattern_text, weights))
        return (10 - checksum % 10) % 10 == int(pattern_text[-1])
