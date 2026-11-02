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
    from .ae_emirates_id_recognizer import AeEmiratesIdRecognizer
    from .credit_card_recognizer import CreditCardRecognizer
    from .in_bank_account_recognizer import InBankAccountRecognizer
    from .in_aadhaar_recognizer import InAadhaarRecognizer
    from .in_driving_license_recognizer import InDrivingLicenseRecognizer
    from .in_ifsc_recognizer import InIfscRecognizer
    from .in_pan_recognizer import InPanRecognizer
    from .in_passport_recognizer import InPassportRecognizer
    from .in_tan_recognizer import InTanRecognizer
    from .in_upi_recognizer import InUpiRecognizer
    from .nrp_recognizer import NrpRecognizer
    from .phone_recognizer import PhoneRecognizer

    return [
        CreditCardRecognizer,
        PhoneRecognizer,
        InAadhaarRecognizer,
        InDrivingLicenseRecognizer,
        InBankAccountRecognizer,
        InIfscRecognizer,
        InPanRecognizer,
        InPassportRecognizer,
        InUpiRecognizer,
        InTanRecognizer,
        AeEmiratesIdRecognizer,
        NrpRecognizer,
    ]


__all__ = [
    "_get_predefined_recognizers",
]
