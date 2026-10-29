import re

import phonenumbers
from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
from presidio_analyzer.predefined_recognizers import PhoneRecognizer as PresidioPhoneRecognizer


class PhoneRecognizer(PresidioPhoneRecognizer):
    DEFAULT_SUPPORTED_REGIONS = (
        *PresidioPhoneRecognizer.DEFAULT_SUPPORTED_REGIONS,
        "IN",
        "CN",
    )
    SCORE = 0.75

    INDIAN_PHONE_PATTERN = re.compile(
        r"(?<![A-Za-z0-9])"
        r"(?:(?:\+?91)[ -]?[6-9](?:[ -]?\d){9}"
        r"|0[ -]?[6-9](?:[ -]?\d){9}"
        r"|[6-9](?:[ -]?\d){9})"
        r"(?![A-Za-z0-9])"
    )

    def analyze(
        self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None
    ) -> list[RecognizerResult]:
        results = super().analyze(text, entities, nlp_artifacts)
        if "PHONE_NUMBER" not in entities:
            return results

        for match in self.INDIAN_PHONE_PATTERN.finditer(text):
            candidate = match.group()
            digits = re.sub(r"\D", "", candidate)
            # A bare 91 country prefix needs "+" before phonenumbers parses it internationally.
            normalized_number = (
                f"+{digits}"
                if digits.startswith("91") and not candidate.lstrip().startswith("+")
                else candidate
            )

            try:
                parsed_number = phonenumbers.parse(normalized_number, "IN")
            except phonenumbers.NumberParseException:
                continue

            if not phonenumbers.is_valid_number_for_region(parsed_number, "IN"):
                continue

            existing_result = next(
                (
                    result
                    for result in results
                    if result.entity_type == "PHONE_NUMBER"
                    and result.start == match.start()
                    and result.end == match.end()
                ),
                None,
            )
            if existing_result:
                existing_result.score = max(existing_result.score, self.SCORE)
                continue

            results.append(
                RecognizerResult(
                    entity_type="PHONE_NUMBER",
                    start=match.start(),
                    end=match.end(),
                    score=self.SCORE,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                        RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                    },
                )
            )

        return EntityRecognizer.remove_duplicates(results)
