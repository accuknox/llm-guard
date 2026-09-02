from __future__ import annotations

from llm_guard.input_scanners.language import Language as InputLanguage, MatchType
from llm_guard.model import Model

from .base import Scanner


class Language(Scanner):
    """
    Language scanner is responsible for determining the language of a given text
    prompt and verifying its validity against a list of predefined languages.

    Setting allowSelectAll=True allows every language the model knows, so no detected
    language is ever a violation.
    """

    def __init__(
        self,
        valid_languages: list[str],
        *,
        allowSelectAll: bool = False,
        model: Model | None = None,
        threshold: float = 0.7,
        match_type: MatchType | str = MatchType.FULL,
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes the Language scanner with a list of valid languages.

        Parameters:
            model (Model, optional): A Model object containing the path to the model and its ONNX equivalent.
            valid_languages (Sequence[str]): A list of valid language codes.
            allowSelectAll (bool): When True, every language the model supports is treated as
                valid, whatever `valid_languages` says, so nothing is flagged. When False (the
                default) the normal flow applies and only languages outside `valid_languages`
                are flagged.
            threshold (float): Minimum confidence score.
            match_type (MatchType): Whether to match the full text or individual sentences. Default is MatchType.FULL.
            use_onnx (bool): Whether to use ONNX for inference. Default is False.
        """

        self._scanner = InputLanguage(
            valid_languages,
            allowSelectAll=allowSelectAll,
            model=model,
            threshold=threshold,
            match_type=match_type,
            use_onnx=use_onnx,
        )

    def scan(
        self,
        prompt: str,
        output: str,
        valid_languages: list[str] | None = None,
        threshold: float | None = None,
        match_type: MatchType | None = None,
        allowSelectAll: bool | None = None,
    ) -> tuple[str, bool, float]:
        return self._scanner.scan(
            output,
            valid_languages=valid_languages,
            threshold=threshold,
            match_type=match_type,
            allowSelectAll=allowSelectAll,
        )
