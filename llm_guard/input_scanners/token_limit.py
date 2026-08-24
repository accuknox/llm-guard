from __future__ import annotations

from typing import TYPE_CHECKING, cast

from llm_guard.util import get_logger, lazy_load_dep

from .base import Scanner

LOGGER = get_logger()

if TYPE_CHECKING:
    import tiktoken


class TokenLimit(Scanner):
    """
    A token limit scanner based on the tiktoken library. It checks if a prompt exceeds a specific token limit.
    """

    def __init__(
        self,
        *,
        limit: int = 4096,
        encoding_name: str = "cl100k_base",
        model_name: str | None = None,
    ) -> None:
        """
        Initializes TokenLimit with a limit, encoding name, and model name.

        Parameters:
            limit (int): Maximum number of tokens allowed in a prompt. Default is 4096.
            encoding_name (str): Encoding model for the tiktoken library. Default is 'cl100k_base'.
            model_name (str): Specific model for the tiktoken encoding. Default is None.
        """

        self._limit = limit

        tiktoken = cast("tiktoken", lazy_load_dep("tiktoken"))
        if not model_name:
            self._encoding = tiktoken.get_encoding(encoding_name)
        else:
            self._encoding = tiktoken.encoding_for_model(model_name)

    def scan(self, prompt: str, limit: int | None = None) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, -1.0

        if limit is None:
            limit = self._limit

        num_tokens = len(self._encoding.encode(prompt))
        if num_tokens < limit:
            LOGGER.debug(
                "Prompt fits the maximum tokens",
                num_tokens=num_tokens,
                threshold=limit,
            )
            return prompt, True, -1.0

        LOGGER.warning(
            "Prompt is too big",
            num_tokens=num_tokens,
            threshold=limit,
        )

        return prompt, False, 1.0
