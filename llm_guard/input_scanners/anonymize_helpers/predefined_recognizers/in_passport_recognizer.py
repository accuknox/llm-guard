from presidio_analyzer import Pattern
from presidio_analyzer.predefined_recognizers import (
    InPassportRecognizer as PresidioInPassportRecognizer,
)


class InPassportRecognizer(PresidioInPassportRecognizer):
    """Recognizes Indian passport numbers written with a dash.

    Presidio already tolerates a space after the third character (``A12 34567``);
    this also accepts the equally common ``A12-34567``.
    """

    PATTERNS = [
        Pattern(
            "PASSPORT",
            r"\b[A-Z][1-9]\d[-\s]?\d{4}[1-9]\b",
            0.1,
        ),
    ]
