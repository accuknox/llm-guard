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
from .span_attribution import SpanDetector

LOGGER = get_logger()

# Accuknox fine-tuned RoBERTa gibberish classifier. Private repo, so loading
# requires an HF token (set HF_TOKEN in the environment; token=True picks it up).
# Binary labels: 0=GIBBERISH, 1=NORMAL. No ONNX export published, so use_onnx
# would fall back to an on-the-fly export.
DEFAULT_MODEL = Model(
    path="Accuknoxtechnologies/gibberish-deberta",
    revision="7fd98078cc3b730acdf9c66a6b0c9ddcfb8b59b0",
    pipeline_kwargs={
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
    },
    tokenizer_kwargs={"token": True},
    kwargs={"token": True},
)

_gibberish_labels = ["GIBBERISH"]


class MatchType(Enum):
    SENTENCE = "sentence"
    FULL = "full"

    def get_inputs(self, prompt: str) -> list[str]:
        if self == MatchType.SENTENCE:
            return split_text_by_sentences(prompt)

        return [prompt]


class Gibberish(Scanner):
    """
    A scanner that detects gibberish text.
    """

    def __init__(
        self,
        *,
        model: Model | None = None,
        threshold: float = 0.97,
        match_type: MatchType | str = MatchType.FULL,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes the Gibberish scanner with a probability threshold for gibberish detection.

        Parameters:
           model (Model, optional): The model object.
           threshold (float): The probability threshold for gibberish detection. Default is 0.7.
           match_type (MatchType): Whether to match the full text or individual sentences. Default is MatchType.FULL.
           use_onnx (bool): Whether to use ONNX instead of PyTorch for inference.
        """
        if isinstance(match_type, str):
            match_type = MatchType(match_type)

        self._threshold = threshold
        self._match_type = match_type

        if model is None:
            model = DEFAULT_MODEL

        tf_tokenizer, tf_model = get_tokenizer_and_model_for_classification(
            model=model,
            use_onnx=use_onnx,
        )

        self._classifier = pipeline(
            task="text-classification",
            model=tf_model,
            tokenizer=tf_tokenizer,
            **model.pipeline_kwargs,
        )

        # Built lazily on the first analyze_spans() call so captum is only
        # required when span attribution is actually used.
        self._span_detector: SpanDetector | None = None

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        # Chunk long inputs so nothing past the model's 512-token window is
        # silently truncated; score every chunk and keep the max.
        inputs = []
        for text in self._match_type.get_inputs(prompt):
            inputs.extend(split_text_to_token_chunks(self._classifier.tokenizer, text))
        if not inputs:
            return prompt, True, -1.0

        highest_score = 0.0
        results_all = self._classifier(inputs)
        LOGGER.debug("Gibberish detection finished", results=results_all)
        for result in results_all:
            score = round(
                (result["score"] if result["label"] in _gibberish_labels else 1 - result["score"]),
                2,
            )

            if score > highest_score:
                highest_score = score

        if highest_score > self._threshold:
            LOGGER.warning(
                "Detected gibberish text",
                score=highest_score,
                threshold=self._threshold,
            )

            return prompt, False, calculate_risk_score(highest_score, self._threshold)

        LOGGER.debug(
            "No gibberish in the text",
            highest_score=highest_score,
            threshold=self._threshold,
        )

        return prompt, True, calculate_risk_score(highest_score, self._threshold)

    def analyze_spans(self, prompt: str) -> list[dict]:
        """Return the character spans of the prompt that drive the GIBBERISH class.

        Meant to be called only after scan() has flagged the prompt (e.g. to
        explain a violation), since it runs an Integrated Gradients pass.

        Returns a list of {"text", "score", "start", "end"} dicts.
        """
        if prompt.strip() == "":
            return []

        if self._span_detector is None:
            model = self._classifier.model
            target_class = model.config.label2id.get("GIBBERISH", 0)
            self._span_detector = SpanDetector(
                model=model,
                tokenizer=self._classifier.tokenizer,
                target_class=target_class,
                classify_threshold=self._threshold,
            )

        return self._span_detector.detect_as_dicts(prompt)
