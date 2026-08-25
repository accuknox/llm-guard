from __future__ import annotations

from llm_guard.input_scanners.code import Code as InputCode
from llm_guard.model import Model

from .base import Scanner


class Code(Scanner):
    """
    A class for scanning if the model output includes code in specific programming languages.

    This class uses the transformers library to detect code snippets in the output of the language model.
    The languages it is configured with are the ones that are *blocked*: code
    detected in any of them is a violation, and code in every other language passes
    through. Selecting every language the model supports (which is also what
    passing no languages means) blocks any code at all, i.e. the scanner behaves
    like the BanCode scanner.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        *,
        model: Model | None = None,
        threshold: float = 0.83,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes an instance of the Code class.

        Parameters:
            model: The model to use for language detection.
            languages: The list of programming languages to block. Code in any of
                them is flagged; code in any other language is allowed. Passing
                every supported language - or None/an empty list, which means the
                same thing - blocks all code (BanCode behaviour).
            threshold: The threshold for the model output to be considered valid.
                Default is 0.83, the default encoder's recommended global operating point.
            use_onnx: Whether to use ONNX for inference. Default is False.

        Raises:
            LLMGuardValidationError: If the languages are not a subset of the
                loaded model's own labels.
        """

        self._scanner = InputCode(
            languages,
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
    ) -> tuple[str, bool, float]:
        return self._scanner.scan(
            output,
            languages=languages,
            threshold=threshold,
        )
