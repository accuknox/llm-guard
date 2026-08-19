from presidio_analyzer import Pattern
from presidio_analyzer.predefined_recognizers import (
    InAadhaarRecognizer as PresidioInAadhaarRecognizer,
)


class InAadhaarRecognizer(PresidioInAadhaarRecognizer):
    """Recognizes Aadhaar numbers written in the usual 4-4-4 grouping.

    Presidio only matches 12 contiguous digits, so the separator stripping it does in
    ``validate_result`` never runs for numbers written as ``2341-2341-2346`` or
    ``2341 2341 2346``. The separator has to be the same one in both positions, which
    keeps things like ``2341-23412346`` from matching.
    """

    PATTERNS = [
        Pattern(
            "AADHAAR (Very Weak)",
            r"\b[0-9]{4}([- :]?)[0-9]{4}\1[0-9]{4}\b",
            0.01,
        ),
    ]
