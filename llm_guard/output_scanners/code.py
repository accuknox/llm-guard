from __future__ import annotations

from llm_guard.input_scanners.code import Code as InputCode
from llm_guard.model import Model

from .base import Scanner


class Code(Scanner):
    """
    A class for scanning if the model output includes code in specific programming languages.

    This class uses the transformers library to detect code snippets in the output of the language model.
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
        Initializes an instance of the Code class.

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
            threshold: The threshold for the model output to be considered valid.
                Default is 0.83, the default encoder's recommended global operating point.
            use_onnx: Whether to use ONNX for inference. Default is False.

        Raises:
            LLMGuardValidationError: If the languages are not a subset of the
                loaded model's own labels.
        """

        self._scanner = InputCode(
            languages,
            allowSelectAll=allowSelectAll,
            model=model,
            threshold=threshold,
            use_onnx=use_onnx,
        )

    def scan(
        self,
        prompt: str,
        output: str,
        languages: list[str] | None = None,
        threshold: float | None = None,
        allowSelectAll: bool | None = None,
    ) -> tuple[str, bool, float]:
        return self._scanner.scan(
            output,
            languages=languages,
            threshold=threshold,
            allowSelectAll=allowSelectAll,
        )
