from typing import List, Optional, Tuple

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer


class AeEmiratesIdRecognizer(PatternRecognizer):
    """Recognizes UAE Emirates ID numbers.

    An Emirates ID is 15 digits, usually written as ``784-YYYY-NNNNNNN-C``: the UAE
    country code 784, the holder's year of birth, a 7 digit serial number and a Luhn
    check digit computed over the first 14 digits. The separator can be a dash, a space
    or nothing, but it has to be the same one in every position, which keeps things like
    ``784-1990 1234567-1`` from matching. Matches that fail the Luhn check are dropped.
    """

    PATTERNS = [
        Pattern(
            "Emirates ID (Weak)",
            r"\b784([- ]?)(?:19|20)\d{2}\1\d{7}\1\d\b",
            0.3,
        ),
    ]

    CONTEXT = [
        "emirates",
        "eid",
        "uae",
        "identity",
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "en",
        supported_entity: str = "AE_EMIRATES_ID",
        replacement_pairs: Optional[List[Tuple[str, str]]] = None,
    ):
        self.replacement_pairs = replacement_pairs or [("-", ""), (" ", "")]
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns or self.PATTERNS,
            context=context or self.CONTEXT,
            supported_language=supported_language,
        )

    def validate_result(self, pattern_text: str) -> bool:
        sanitized_value = EntityRecognizer.sanitize_value(pattern_text, self.replacement_pairs)
        return (
            len(sanitized_value) == 15
            and sanitized_value.isdigit()
            and self.luhn_check_digit(sanitized_value[:-1]) == int(sanitized_value[-1])
        )

    @staticmethod
    def luhn_check_digit(payload: str) -> int:
        """Returns the Luhn check digit that has to be appended to ``payload``."""
        total = 0
        # Walking from the right, every other digit starting with the rightmost one is
        # doubled, because the check digit will take the rightmost position once appended.
        for i, char in enumerate(reversed(payload)):
            digit = int(char)
            if i % 2 == 0:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return (10 - total % 10) % 10
