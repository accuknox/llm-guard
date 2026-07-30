from __future__ import annotations

import re

from llm_guard.exception import LLMGuardValidationError
from llm_guard.model import Model
from llm_guard.transformers_helpers import get_tokenizer_and_model_for_classification, pipeline
from llm_guard.util import calculate_risk_score, get_logger

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
# No ONNX export published, so use_onnx would fall back to an on-the-fly export.
CODE_LANG_ENCODER_V1 = Model(
    path="Accuknoxtechnologies/CodeLanguage-codebert-base-Encoder-v1",
    revision="efea2d9c77eba33a9ba1718c6d027ce1bd1a2f8c",
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

DEFAULT_MODEL = CODE_LANG_ENCODER_V1

# The 25 languages the default model identifies (its config.id2label). Validation
# in __init__ runs against the loaded model's own labels, so a custom model with a
# different label set is supported automatically; this list is informational.
SUPPORTED_LANGUAGES = [
    "Python",
    "JavaScript",
    "Java",
    "C",
    "C++",
    "C#",
    "Go",
    "Rust",
    "Kotlin",
    "Swift",
    "Ruby",
    "R",
    "Scala",
    "Perl",
    "Lua",
    "Bash",
    "PowerShell",
    "Batch",
    "SQL",
    "Dockerfile",
    "YAML",
    "Makefile",
    "Terraform",
    "AWK",
    "jq",
]


class Code(Scanner):
    """
    A class for scanning if the prompt includes code in specific programming languages.

    This class uses the transformers library to detect code snippets in the output of the language model.
    It can be configured to allow or block specific programming languages.
    """

    def __init__(
        self,
        languages: list[str],
        *,
        model: Model | None = None,
        is_blocked: bool = True,
        threshold: float = 0.5,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes Code with the allowed and denied languages.

        Parameters:
            model: The model to use for language detection.
            languages: The list of programming languages to allow or deny.
            is_blocked: Whether the languages are blocked or allowed. Default is True.
            threshold: The threshold for the risk score. Default is 0.5.
            use_onnx: Whether to use ONNX for inference. Default is False.

        Raises:
            LLMGuardValidationError: If the languages are not a subset of SUPPORTED_LANGUAGES.
        """
        self._languages = languages
        self._is_blocked = is_blocked
        self._threshold = threshold

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
        supported = set(self._pipeline.model.config.id2label.values())
        if not set(languages).issubset(supported):
            raise LLMGuardValidationError(f"Languages must be a subset of {sorted(supported)}")

        self._fenced_code_regex = re.compile(r"```(?:[a-zA-Z0-9]*\n)?(.*?)```", re.DOTALL)
        self._inline_code_regex = re.compile(r"`(.*?)`")

        # Built lazily on the first analyze_spans() call so captum is only
        # required when span attribution is actually used.
        self._span_detector: SpanDetector | None = None

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
        is_blocked: bool | None = None,
        threshold: float | None = None,
    ) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        languages_config = languages if languages is not None else self._languages
        if is_blocked is None:
            is_blocked = self._is_blocked
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

        # Only check when the code is detected
        results = self._pipeline(code_blocks)
        for code_block, results_languages in zip(code_blocks, results):
            LOGGER.debug(
                "Detected languages in the code",
                languages=results_languages,
                code_block=code_block,
            )

            for language in results_languages:
                score = round(language["score"], 2)

                if score < threshold or language["label"] not in languages_config:
                    continue

                if is_blocked:
                    LOGGER.warning(
                        "Language is not allowed",
                        language_name=language["label"],
                        score=score,
                    )
                    return prompt, False, calculate_risk_score(score, threshold)

                if not is_blocked:
                    LOGGER.debug(
                        "Language is allowed",
                        language_name=language["label"],
                        score=score,
                    )
                    return prompt, True, calculate_risk_score(score, threshold)

        if is_blocked:
            LOGGER.debug("No blocked languages detected")
            return prompt, True, -1.0

        LOGGER.warning("No allowed languages detected")
        return prompt, False, 1.0

    def analyze_spans(self, prompt: str) -> list[dict]:
        """Return the character spans of the prompt that drive the violation.

        Meant to be called only after scan() has flagged the prompt (e.g. to
        explain a violation), since it runs an Integrated Gradients pass. The
        default encoder is multi-label (independent sigmoid per language), so the
        detector attributes the highest-scoring candidate language per chunk.

        The candidate languages depend on the policy: in block mode they are the
        configured (banned) languages; in allow mode they are every other language
        the model knows (whose presence would be a violation).

        Returns a list of {"text", "score", "start", "end"} dicts.
        """
        if prompt.strip() == "":
            return []

        if self._span_detector is None:
            model = self._pipeline.model
            label2id = model.config.label2id
            if self._is_blocked:
                targets = [label2id[lang] for lang in self._languages if lang in label2id]
            else:
                allowed = set(self._languages)
                targets = [idx for lang, idx in label2id.items() if lang not in allowed]
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
