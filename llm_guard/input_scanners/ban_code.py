from __future__ import annotations

import re

from llm_guard.model import Model
from llm_guard.transformers_helpers import get_tokenizer_and_model_for_classification, pipeline
from llm_guard.util import (
    calculate_risk_score,
    get_logger,
    remove_markdown,
    split_text_to_token_chunks,
)

from .base import Scanner
from .span_attribution import SpanDetector

LOGGER = get_logger()

# Accuknox fine-tuned CodeBERT (RoBERTa) classifier. Private repo, so loading
# requires an HF token (set HF_TOKEN in the environment; token=True picks it up).
# Labels: 0=CODE, 1=NL. No ONNX export published, so use_onnx would fall back to
# an on-the-fly export.
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


class BanCode(Scanner):
    """
    A scanner that detects if input is code and blocks it.
    """

    def __init__(
        self,
        *,
        model: Model | None = None,
        threshold: float = 0.97,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes the BanCode scanner.

        Parameters:
           model (Model, optional): The model object.
           threshold (float): The probability threshold. Default is 0.97.
           use_onnx (bool): Whether to use ONNX instead of PyTorch for inference.
        """

        self._threshold = threshold
        if model is None:
            model = MODEL_SM

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

        # Hack: Improve accuracy
        new_prompt = remove_markdown(prompt)  # Remove markdown
        new_prompt = re.sub(r"\d+\.\s+|[-*•]\s+", "", new_prompt)  # Remove list markers
        new_prompt = re.sub(r"\d+", "", new_prompt)  # Remove numbers
        new_prompt = re.sub(r'\.(?!\d)(?=[\s\'"“”‘’)\]}]|$)', "", new_prompt)  # Remove periods

        # Chunk long inputs so nothing past the model's 512-token window is
        # silently truncated; score every chunk and keep the max.
        chunks = split_text_to_token_chunks(self._classifier.tokenizer, new_prompt)
        if not chunks:
            return prompt, True, -1.0

        score = 0.0
        for result in self._classifier(chunks):
            chunk_score = round(
                result["score"] if result["label"] in "CODE" else 1 - result["score"],
                2,
            )
            score = max(score, chunk_score)

        if score > threshold:
            LOGGER.warning(
                "Detected code in the text",
                score=score,
                threshold=threshold,
                text=new_prompt,
            )

            return prompt, False, calculate_risk_score(score, threshold)

        LOGGER.debug(
            "No code detected in the text",
            score=score,
            threshold=threshold,
            text=new_prompt,
        )

        return prompt, True, calculate_risk_score(score, threshold)

    def analyze_spans(self, prompt: str) -> list[dict]:
        """Return the character spans of the prompt that drive the CODE class.

        Meant to be called only after scan() has flagged the prompt (e.g. to
        explain a violation), since it runs an Integrated Gradients pass. Note:
        attribution runs on the original prompt (not the markdown/number-stripped
        text scan() uses) so offsets map back to the caller's input.

        Returns a list of {"text", "score", "start", "end"} dicts.
        """
        if prompt.strip() == "":
            return []

        if self._span_detector is None:
            model = self._classifier.model
            target_class = model.config.label2id.get("CODE", 0)
            self._span_detector = SpanDetector(
                model=model,
                tokenizer=self._classifier.tokenizer,
                target_class=target_class,
                classify_threshold=self._threshold,
            )

        return self._span_detector.detect_as_dicts(prompt)
