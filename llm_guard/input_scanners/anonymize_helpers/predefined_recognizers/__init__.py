from typing import Callable

from presidio_analyzer import EntityRecognizer


def _get_predefined_recognizers(language: str) -> list[Callable[..., EntityRecognizer]]:
    if language == "zh":
        from .zh import CryptoRecognizer, EmailRecognizer, IpRecognizer, PhoneRecognizer

        return [
            CryptoRecognizer,
            PhoneRecognizer,
            EmailRecognizer,
            IpRecognizer,
        ]
    from .credit_card_recognizer import CreditCardRecognizer
    from .in_aadhaar_recognizer import InAadhaarRecognizer
    from .in_passport_recognizer import InPassportRecognizer

    return [CreditCardRecognizer, InAadhaarRecognizer, InPassportRecognizer]


__all__ = [
    "_get_predefined_recognizers",
]
