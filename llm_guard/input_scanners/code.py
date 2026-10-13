from __future__ import annotations

import re

from llm_guard.exception import LLMGuardValidationError
from llm_guard.model import Model
from llm_guard.transformers_helpers import get_tokenizer_and_model_for_classification, pipeline
from llm_guard.util import calculate_risk_score, get_logger, split_text_to_token_chunks

from .base import Scanner
from .span_attribution import SpanDetector

LOGGER = get_logger()

# Legacy single-label softmax classifier. Kept for backward compatibility; pass
# it explicitly via model= to use it instead of the default encoder.
PHILOMATH_MODEL = Model(
    path="philomath-1209/programming-language-identification",
    revision="9090d38e7333a2c6ff00f154ab981a549842c20f",
    onnx_path="philomath-1209/programming-language-identification",
    onnx_revision="9090d38e7333a2c6ff00f154ab981a549842c20f",
    onnx_subfolder="onnx",
    pipeline_kwargs={
        "top_k": None,
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
    },
)

# Accuknox multi-label code-language encoder, fine-tuned from microsoft/codebert-base.
# Private repo, so loading needs an HF token (set HF_TOKEN; token=True picks it up).
# problem_type=multi_label_classification => the pipeline applies an independent
# sigmoid per language, and top_k=None returns every language with its own score.
# Recommended operating point is a global threshold of 0.83 (see the model's
# thresholds.json). An int8 ONNX export lives under onnx-int8/, but no plain ONNX
# export is published, so use_onnx would fall back to an on-the-fly export.
CODE_IDENTIFICATION_ENCODER_V1 = Model(
    path="Accuknoxtechnologies/Code-Identification-Encoder-v1",
    revision="2104a96e1a5cfc1b98eb8fea759d8560e331b097",
    pipeline_kwargs={
        "top_k": None,
        "function_to_apply": "sigmoid",
        "return_token_type_ids": False,
        "max_length": 512,
        "truncation": True,
    },
    tokenizer_kwargs={"token": True},
    kwargs={"token": True},
)

DEFAULT_MODEL = CODE_IDENTIFICATION_ENCODER_V1

# The languages the default model identifies (its config.id2label). Validation in
# __init__ runs against the loaded model's own labels, so a custom model with a
# different label set is supported automatically; this list is informational.
SUPPORTED_LANGUAGES = [
    "AWK",
    "Bash",
    "Batch",
    "C",
    "C#",
    "C++",
    "Dockerfile",
    "Go",
    "Java",
    "JavaScript",
    "Kotlin",
    "Lua",
    "Makefile",
    "Perl",
    "PowerShell",
    "Python",
    "R",
    "Ruby",
    "Rust",
    "SQL",
    "Scala",
    "Swift",
    "Terraform",
    "jq",
]


class Code(Scanner):
    """
    A class for scanning if the prompt includes code in specific programming languages.

    This class uses the transformers library to detect code snippets in the prompt.
    The languages it is configured with are the ones that are *allowed*: code
    detected in any of them passes through, and code in every other language the
    model supports is a violation. Selecting no languages blocks any code at all,
    i.e. the scanner behaves like the BanCode scanner. Setting allowSelectAll=True
    allows every supported language regardless of the languages given, so nothing
    is flagged.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        *,
        allowSelectAll: bool = False,
        model: Model | None = None,
        threshold: float = 0.83,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes Code with the allowed languages.

        Parameters:
            model: The model to use for language detection.
            languages: The list of programming languages to allow. Code in any of
                them passes; code in any other supported language is flagged.
                Passing None or an empty list allows nothing, so all code is
                blocked (BanCode behaviour).
            allowSelectAll: When True, every language the model supports is
                allowed, whatever `languages` says, so nothing is flagged. When
                False (the default) the normal flow applies and only `languages`
                is allowed.
            threshold: The threshold for the risk score. Default is 0.83, the
                default encoder's recommended global operating point.
            use_onnx: Whether to use ONNX for inference. Default is False.

        Raises:
            LLMGuardValidationError: If the languages are not a subset of the
                loaded model's own labels.
        """
        self._threshold = threshold
        self._allow_select_all = allowSelectAll

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

        # Validate the requested languages against the loaded model's own labels,
        # so any model (the default encoder or a custom one) with a different label
        # set works without editing a hardcoded list.
        self._supported_languages = set(self._pipeline.model.config.id2label.values())
        if not set(languages or []).issubset(self._supported_languages):
            raise LLMGuardValidationError(
                f"Languages must be a subset of {sorted(self._supported_languages)}"
            )

        # No languages given means nothing is allowed, so the scanner blocks all code.
        self._languages = set(languages or [])

        self._fenced_code_regex = re.compile(r"```(?:[a-zA-Z0-9]*\n)?(.*?)```", re.DOTALL)
        self._inline_code_regex = re.compile(r"`(.*?)`")

        # Built lazily on the first analyze_spans() call so captum is only
        # required when span attribution is actually used.
        self._span_detector: SpanDetector | None = None

    def _blocked_languages(self, allowed: set[str]) -> set[str]:
        """Every language the model knows that is not on the given allow-list."""
        return self._supported_languages - allowed

    def _extract_code_blocks(self, markdown: str) -> list[str]:
        # Extract fenced code blocks (between triple backticks)
        fenced_code_blocks = [
            block.strip() for block in self._fenced_code_regex.findall(markdown) if block.strip()
        ]

        # Extract inline code (between single backticks)
        inline_code = [
            code.strip()
            for code in self._inline_code_regex.findall(markdown)
            if code.strip() and any(char in code for char in "{}[]()=+-*/<>!")
        ]

        return fenced_code_blocks + inline_code

    def scan(
        self,
        prompt: str,
        languages: list[str] | None = None,
        threshold: float | None = None,
        allowSelectAll: bool | None = None,
    ) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        if allowSelectAll is None:
            allowSelectAll = self._allow_select_all

        # "Select all" overrides the allow-list entirely: every supported
        # language is allowed, so nothing is a violation. An explicit empty list
        # allows nothing; only None falls back to the configured languages.
        if allowSelectAll:
            allowed = set(self._supported_languages)
        elif languages is None:
            allowed = self._languages
        else:
            allowed = set(languages)
        blocked = self._blocked_languages(allowed)

        if threshold is None:
            threshold = self._threshold

        # Try to extract code snippets from Markdown
        code_blocks = self._extract_code_blocks(prompt)
        if len(code_blocks) == 0:
            LOGGER.debug(
                "No Markdown code blocks found in the output. Using the whole input as code."
            )
            code_blocks = [prompt]

        LOGGER.debug("Code blocks found in the output", code_blocks=code_blocks)

        # Chunk each block so nothing past the model's 512-token window is
        # silently truncated; every chunk is classified.
        chunks = []
        for block in code_blocks:
            chunks.extend(split_text_to_token_chunks(self._pipeline.tokenizer, block))
        if not chunks:
            return prompt, True, -1.0

        # The default encoder is multi-label, so a single chunk can report several
        # languages, each with its own score. Look at every language the model
        # detects above the threshold and keep the strongest violation: a detected
        # language that is not on the allow-list. Code in an allowed language,
        # and text with no code at all (no language above the threshold), has no
        # violation and passes. When the allow-list is empty, any code is a
        # violation (BanCode behaviour).
        results = self._pipeline(chunks)
        violating_language: str | None = None
        violating_score = 0.0
        for code_block, results_languages in zip(chunks, results):
            LOGGER.debug(
                "Detected languages in the code",
                languages=results_languages,
                code_block=code_block,
            )

            if isinstance(results_languages, dict):
                results_languages = [results_languages]

            for language in results_languages:
                score = round(language["score"], 2)
                if score < threshold:
                    continue

                label = language["label"]
                if label in blocked and score > violating_score:
                    violating_language = label
                    violating_score = score

        bans_all_code = not allowed
        if violating_language is not None:
            LOGGER.warning(
                "Code is not allowed" if bans_all_code else "Language is not allowed",
                language_name=violating_language,
                score=violating_score,
            )
            return prompt, False, calculate_risk_score(violating_score, threshold)

        LOGGER.debug("No code detected" if bans_all_code else "No blocked languages detected")
        return prompt, True, -1.0

    def analyze_spans(self, prompt: str) -> list[dict]:
        """Return the character spans of the prompt that drive the violation.

        Meant to be called only after scan() has flagged the prompt (e.g. to
        explain a violation), since it runs an Integrated Gradients pass. The
        default encoder is multi-label (independent sigmoid per language), so the
        detector attributes the highest-scoring candidate language per chunk.

        The candidate languages are the blocked ones - every language the model
        knows that is not on the configured allow-list - which is all of them
        when the scanner blocks all code.

        Returns a list of {"text", "score", "start", "end"} dicts.
        """
        if prompt.strip() == "":
            return []

        if self._span_detector is None:
            model = self._pipeline.model
            label2id = model.config.label2id
            candidates = (
                set() if self._allow_select_all else self._blocked_languages(self._languages)
            )
            targets = [label2id[lang] for lang in candidates if lang in label2id]
            if not targets:
                targets = list(range(model.config.num_labels))

            self._span_detector = SpanDetector(
                model=model,
                tokenizer=self._pipeline.tokenizer,
                target_class=targets,
                multi_label=True,
                classify_threshold=self._threshold,
            )

        return self._span_detector.detect_as_dicts(prompt)
