from __future__ import annotations

from enum import Enum

from llm_guard.model import Model
from llm_guard.transformers_helpers import get_tokenizer_and_model_for_classification, pipeline
from llm_guard.util import (
    calculate_risk_score,
    get_logger,
    split_text_by_sentences,
    split_text_to_token_chunks,
)

from .base import Scanner

LOGGER = get_logger()

DEFAULT_MODEL = Model(
    path="papluca/xlm-roberta-base-language-detection",
    revision="9865598389ca9d95637462f743f683b51d75b87b",
    onnx_path="ProtectAI/xlm-roberta-base-language-detection-onnx",
    onnx_revision="dce2fa14a0dc61b6f889537e9ad4fccf083b22bd",
    pipeline_kwargs={
        "top_k": None,
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
    },
)


class MatchType(Enum):
    SENTENCE = "sentence"
    FULL = "full"

    def get_inputs(self, prompt: str) -> list[str]:
        if self == MatchType.SENTENCE:
            return split_text_by_sentences(prompt)

        return [prompt]


class Language(Scanner):
    """
    A Scanner subclass responsible for determining the language of a given text
    prompt and verifying its validity against a list of predefined languages.

    Note: when no languages are detected above the threshold, the prompt is considered valid.

    Setting allowSelectAll=True allows every language the model knows, so no detected
    language is ever a violation.
    """

    def __init__(
        self,
        valid_languages: list[str],
        *,
        allowSelectAll: bool = False,
        model: Model | None = None,
        threshold: float = 0.6,
        match_type: MatchType | str = MatchType.FULL,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes the Language scanner with a list of valid languages.

        Parameters:
            model (Model, optional): A Model object containing the path to the model and its ONNX equivalent.
            valid_languages (Sequence[str]): A list of valid language codes in ISO 639-1.
            allowSelectAll (bool): When True, every language the model supports is treated as
                valid, whatever `valid_languages` says, so nothing is flagged. When False (the
                default) the normal flow applies and only languages outside `valid_languages`
                are flagged.
            threshold (float): Minimum confidence score.
            match_type (MatchType): Whether to match the full text or individual sentences. Default is MatchType.FULL.
            use_onnx (bool): Whether to use ONNX for inference. Default is False.
        """
        if isinstance(match_type, str):
            match_type = MatchType(match_type)

        self._threshold = threshold
        self._valid_languages = valid_languages
        self._allow_select_all = allowSelectAll
        self._match_type = match_type

        if model is None:
            model = DEFAULT_MODEL

        tf_tokenizer, tf_model = get_tokenizer_and_model_for_classification(
            model=model,
            use_onnx=use_onnx,
        )

        self._pipeline = pipeline(
            task="text-classification",
            model=tf_model,
            tokenizer=tf_tokenizer,
            **model.pipeline_kwargs,
        )

        # The languages the loaded model can actually detect; used when
        # allowSelectAll marks every language as valid.
        self._supported_languages = list(self._pipeline.model.config.id2label.values())

    def scan(
        self,
        prompt: str,
        valid_languages: list[str] | None = None,
        threshold: float | None = None,
        match_type: MatchType | None = None,
        allowSelectAll: bool | None = None,
    ) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        if allowSelectAll is None:
            allowSelectAll = self._allow_select_all

        if allowSelectAll:
            # "Select all" allows every language the model knows, so nothing is flagged.
            valid_languages = self._supported_languages
        elif valid_languages is None:
            valid_languages = self._valid_languages
        if threshold is None:
            threshold = self._threshold
        if match_type is None:
            match_type = self._match_type
        if isinstance(match_type, str):
            match_type = MatchType(match_type)

        # Chunk long inputs so nothing past the model's 512-token window is
        # silently truncated; score every chunk.
        inputs = []
        for text in match_type.get_inputs(prompt):
            inputs.extend(split_text_to_token_chunks(self._pipeline.tokenizer, text))
        if not inputs:
            return prompt, True, -1.0

        results_all = self._pipeline(inputs)
        for result_chunk in results_all:
            languages_above_threshold = [
                result["label"] for result in result_chunk if result["score"] > threshold
            ]

            highest_score = max([result["score"] for result in result_chunk])

            # Check if any of the languages above threshold are not valid
            if len(set(languages_above_threshold) - set(valid_languages)) > 0:
                LOGGER.warning(
                    "Languages are found with high confidence",
                    languages=languages_above_threshold,
                )

                return (
                    prompt,
                    False,
                    calculate_risk_score(highest_score, threshold),
                )

        LOGGER.debug("Only valid languages are found in the text.")

        return prompt, True, -1.0
