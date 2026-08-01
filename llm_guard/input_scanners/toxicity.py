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

DEFAULT_MODEL = Model(
    path="unitary/unbiased-toxic-roberta",
    revision="36295dd80b422dc49f40052021430dae76241adc",
    onnx_path="ProtectAI/unbiased-toxic-roberta-onnx",
    onnx_revision="34480fa958f6657ad835c345808475755b6974a7",
    pipeline_kwargs={
        "padding": True,
        "top_k": None,
        "function_to_apply": "sigmoid",
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
        "batch_size": 4,
    },
)

_toxic_labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


class MatchType(Enum):
    SENTENCE = "sentence"
    FULL = "full"

    def get_inputs(self, prompt: str) -> list[str]:
        if self == MatchType.SENTENCE:
            return split_text_by_sentences(prompt)

        return [prompt]


class Toxicity(Scanner):
    """
    A toxicity scanner that uses a pretrained Hugging Face model to assess the toxicity of a given text.

    If the toxicity score is less than a predefined threshold, the text is considered non-toxic. Otherwise, it is
    considered toxic.
    """

    def __init__(
        self,
        *,
        model: Model | None = None,
        threshold: float = 0.5,
        match_type: MatchType | str = MatchType.FULL,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes Toxicity with a threshold for toxicity.

        Parameters:
           model (Model, optional): Path to the model. Default is None.
           threshold (float): Threshold for toxicity. Default is 0.5.
           match_type (MatchType): Whether to match the full text or individual sentences. Default is MatchType.FULL.
           use_onnx (bool): Whether to use ONNX for inference. Default is False.
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

        self._pipeline = pipeline(
            task="text-classification",
            model=tf_model,
            tokenizer=tf_tokenizer,
            **model.pipeline_kwargs,
        )

        # Built lazily on the first analyze_spans() call so captum is only
        # required when span attribution is actually used.
        self._span_detector: SpanDetector | None = None

    def scan(
        self,
        prompt: str,
        threshold: float | None = None,
        match_type: MatchType | None = None,
    ) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        if threshold is None:
            threshold = self._threshold
        if match_type is None:
            match_type = self._match_type

        # Chunk long inputs so nothing past the model's 512-token window is
        # silently truncated; score every chunk and keep the max.
        inputs = []
        for text in match_type.get_inputs(prompt):
            inputs.extend(split_text_to_token_chunks(self._pipeline.tokenizer, text))
        if not inputs:
            return prompt, True, -1.0

        highest_toxicity_score = 0.0
        toxicity_above_threshold = []
        results_all = self._pipeline(inputs)
        for results_chunk in results_all:
            for result in results_chunk:
                if result["label"] not in _toxic_labels:
                    continue

                if result["score"] > threshold:
                    toxicity_above_threshold.append(result)

                if result["score"] > highest_toxicity_score:
                    highest_toxicity_score = result["score"]

        if len(toxicity_above_threshold) > 0:
            LOGGER.warning("Detected toxicity in the text", results=toxicity_above_threshold)

            return (
                prompt,
                False,
                calculate_risk_score(highest_toxicity_score, threshold),
            )

        LOGGER.debug("Not toxicity found in the text", results=results_all)

        return (
            prompt,
            True,
            calculate_risk_score(highest_toxicity_score, threshold),
        )

    def analyze_spans(self, prompt: str) -> list[dict]:
        """Return the character spans of the prompt that drive the toxic labels.

        Meant to be called only after scan() has flagged the prompt (e.g. to
        explain a violation), since it runs an Integrated Gradients pass. This is
        a multi-label (sigmoid) model, so attribution targets the highest-scoring
        toxic label in each chunk.

        Returns a list of {"text", "score", "start", "end"} dicts.
        """
        if prompt.strip() == "":
            return []

        if self._span_detector is None:
            model = self._pipeline.model
            label2id = model.config.label2id
            target_labels = [label2id[label] for label in _toxic_labels if label in label2id]
            self._span_detector = SpanDetector(
                model=model,
                tokenizer=self._pipeline.tokenizer,
                target_class=target_labels,
                multi_label=True,
                classify_threshold=self._threshold,
            )

        return self._span_detector.detect_as_dicts(prompt)
