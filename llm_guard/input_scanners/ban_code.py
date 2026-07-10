from __future__ import annotations

import re

from llm_guard.model import Model
from llm_guard.transformers_helpers import get_tokenizer_and_model_for_classification, pipeline
from llm_guard.util import calculate_risk_score, get_logger, remove_markdown

from .base import Scanner

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

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        # Hack: Improve accuracy
        new_prompt = remove_markdown(prompt)  # Remove markdown
        new_prompt = re.sub(r"\d+\.\s+|[-*•]\s+", "", new_prompt)  # Remove list markers
        new_prompt = re.sub(r"\d+", "", new_prompt)  # Remove numbers
        new_prompt = re.sub(r'\.(?!\d)(?=[\s\'"“”‘’)\]}]|$)', "", new_prompt)  # Remove periods

        result = self._classifier(new_prompt)[0]
        score = round(
            result["score"] if result["label"] in "CODE" else 1 - result["score"],
            2,
        )

        if score > self._threshold:
            LOGGER.warning(
                "Detected code in the text",
                score=score,
                threshold=self._threshold,
                text=new_prompt,
            )

            return prompt, False, calculate_risk_score(score, self._threshold)

        LOGGER.debug(
            "No code detected in the text",
            score=score,
            threshold=self._threshold,
            text=new_prompt,
        )

        return prompt, True, calculate_risk_score(score, self._threshold)
