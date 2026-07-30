from __future__ import annotations

from llm_guard.model import Model
from llm_guard.transformers_helpers import get_tokenizer_and_model_for_classification, pipeline
from llm_guard.util import (
    calculate_risk_score,
    get_logger,
    split_text_to_token_chunks,
)

from .base import Scanner
from .code import CODE_LANG_ENCODER_V1
from .span_attribution import SpanDetector

LOGGER = get_logger()

# Legacy binary CODE/NL classifier (labels 0=CODE, 1=NL). Kept for backward
# compatibility; pass it explicitly via model= to use it instead of the default
# encoder. No ONNX export published.
MODEL_SM = Model(
    path="Accuknoxtechnologies/codenl-codebert-banCode",
    revision="c94bc0557860ec2ae9d5786dab29080987ba2abe",
    pipeline_kwargs={
        "max_length": 512,
        "truncation": True,
        "return_token_type_ids": False,
    },
    tokenizer_kwargs={"token": True},
    kwargs={"token": True},
)

# Default: the shared Accuknox multi-label code-language encoder. It emits an
# independent sigmoid per programming language; any language above the threshold
# means code is present, so the prompt is blocked.
DEFAULT_MODEL = CODE_LANG_ENCODER_V1


class BanCode(Scanner):
    """
    A scanner that detects if input is code and blocks it.
    """

    def __init__(
        self,
        *,
        model: Model | None = None,
        threshold: float = 0.5,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes the BanCode scanner.

        Parameters:
           model (Model, optional): The model object. Defaults to the multi-label
               code-language encoder.
           threshold (float): The probability threshold. Default is 0.5, matching
               the default encoder's recommended operating point.
           use_onnx (bool): Whether to use ONNX instead of PyTorch for inference.
        """

        self._threshold = threshold
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

    def scan(self, prompt: str, threshold: float | None = None) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        if threshold is None:
            threshold = self._threshold

        # Chunk long inputs so nothing past the model's 512-token window is
        # silently truncated; score every chunk and keep the max.
        chunks = split_text_to_token_chunks(self._classifier.tokenizer, prompt)
        if not chunks:
            return prompt, True, -1.0

        score = 0.0
        for preds in self._classifier(chunks):
            # Multi-label model: `preds` is a list of {label, score} across every
            # language. "Code present" = the strongest language signal in the
            # chunk. (A single-label model returns one dict; guard for that.)
            if isinstance(preds, dict):
                preds = [preds]
            chunk_score = round(max(pred["score"] for pred in preds), 2)
            score = max(score, chunk_score)

        if score > threshold:
            LOGGER.warning(
                "Detected code in the text",
                score=score,
                threshold=threshold,
            )

            return prompt, False, calculate_risk_score(score, threshold)

        LOGGER.debug(
            "No code detected in the text",
            score=score,
            threshold=threshold,
        )

        return prompt, True, calculate_risk_score(score, threshold)

    def analyze_spans(self, prompt: str) -> list[dict]:
        """Return the character spans of the prompt that drive the code detection.

        Meant to be called only after scan() has flagged the prompt (e.g. to
        explain a violation), since it runs an Integrated Gradients pass. Any code
        language counts as a violation, so the detector attributes the
        highest-scoring language per chunk.

        Returns a list of {"text", "score", "start", "end"} dicts.
        """
        if prompt.strip() == "":
            return []

        if self._span_detector is None:
            model = self._classifier.model
            self._span_detector = SpanDetector(
                model=model,
                tokenizer=self._classifier.tokenizer,
                target_class=list(range(model.config.num_labels)),
                multi_label=True,
                classify_threshold=self._threshold,
            )

        return self._span_detector.detect_as_dicts(prompt)
