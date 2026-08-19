from typing import List, Optional, Tuple

from presidio_analyzer import Pattern
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer as PresidioCreditCardRecognizer,
)


class CreditCardRecognizer(PresidioCreditCardRecognizer):
    """Recognizes card numbers whose groups are separated by dots.

    Presidio handles dashes and spaces but not ``4111.1111.1111.1111``. The dot is
    added to the pattern and to the replacement pairs so the Luhn check in
    ``validate_result`` still sees a bare digit string.
    """

    PATTERNS = [
        Pattern(
            "All Credit Cards (weak)",
            r"\b((4\d{3})|(5[0-5]\d{2})|(6\d{3})|(1\d{3})|(3\d{3}))[-. ]?(\d{3,4})[-. ]?(\d{3,4})[-. ]?(\d{3,5})\b",  # noqa: E501
            0.3,
        ),
    ]

    def __init__(
        self,
        *args,
        replacement_pairs: Optional[List[Tuple[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            replacement_pairs=replacement_pairs or [("-", ""), (" ", ""), (".", "")],
            **kwargs,
        )
