from __future__ import annotations

from llm_guard.input_scanners.code import Code as InputCode
from llm_guard.model import Model

from .base import Scanner


class Code(Scanner):
    """
    A class for scanning if the model output includes code in specific programming languages.

    This class uses the transformers library to detect code snippets in the output of the language model.
    It can be configured to allow or deny specific programming languages.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        *,
        model: Model | None = None,
        is_blocked: bool = False,
        ban_all_code: bool = False,
        threshold: float = 0.83,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes an instance of the Code class.

        Parameters:
            model: The model to use for language detection.
            languages: The list of programming languages to allow or deny. Ignored
                when ban_all_code is enabled; optional in that case.
            is_blocked: Whether the languages are blocked or allowed. Default is False
                (allow-list: only the given languages pass, every other code language is blocked).
            ban_all_code: BanCode mode. When True, any output containing code in any
                language the model recognises is flagged, and `languages` /
                `is_blocked` are ignored. Default is False.
            threshold: The threshold for the model output to be considered valid.
                Default is 0.83, the default encoder's recommended global operating point.
            use_onnx: Whether to use ONNX for inference. Default is False.

        Raises:
            ValueError: If both 'allowed' and 'denied' lists are provided or if both are empty.
        """

        self._scanner = InputCode(
            languages,
            model=model,
            is_blocked=is_blocked,
            ban_all_code=ban_all_code,
            threshold=threshold,
            use_onnx=use_onnx,
        )

    def scan(
        self,
        prompt: str,
        output: str,
        languages: list[str] | None = None,
        is_blocked: bool | None = None,
        ban_all_code: bool | None = None,
        threshold: float | None = None,
    ) -> tuple[str, bool, float]:
        return self._scanner.scan(
            output,
            languages=languages,
            is_blocked=is_blocked,
            ban_all_code=ban_all_code,
            threshold=threshold,
        )
