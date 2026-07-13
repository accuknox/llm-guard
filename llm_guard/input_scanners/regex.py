import re
from enum import Enum
from typing import List, Pattern

from presidio_anonymizer.core.text_replace_builder import TextReplaceBuilder

from llm_guard.util import get_logger

from .base import Scanner

LOGGER = get_logger()


class MatchType(Enum):
    SEARCH = "search"
    FULL_MATCH = "fullmatch"
    ALL = "all"

    def match(self, pattern: Pattern[str], text: str) -> List[re.Match[str]]:
        if self.value == "all":
            return list(pattern.finditer(text))[::-1]  # Reverse order to avoid index issues

        m = None
        if self.value == "search":
            m = pattern.search(text)

        if self.value == "fullmatch":
            m = pattern.fullmatch(text)

        if m is None:
            return []

        return [m]


class Regex(Scanner):
    """
    A class used to detect patterns in the output of a language model using regular expressions.

    This class relies on the list of regular expressions provided by the user. If any of the patterns
    matches the output, the output is considered invalid. It is also possible to redact the output.
    """

    def __init__(
        self,
        patterns: list[str],
        *,
        is_blocked: bool = True,
        match_type: MatchType | str = MatchType.ALL,
        redact: bool = True,
    ) -> None:
        """
        Initializes an instance of the Regex class.

        Parameters:
            patterns (Sequence[str]): A list of regular expressions to use for pattern matching.
            is_blocked (bool): Whether the patterns are blocked or allowed.
            match_type (str): The type of match to use.
            redact (bool): Whether to redact the output or not.

        Raises:
            ValueError: If no patterns are provided or both good and bad patterns are provided.
        """
        if isinstance(match_type, str):
            match_type = MatchType(match_type)

        self._patterns = []
        for pattern in patterns:
            self._patterns.append(re.compile(pattern))

        self._match_type = match_type
        self._is_blocked = is_blocked
        self._redact = redact

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        text_replace_builder = TextReplaceBuilder(original_text=prompt)
        for pattern in self._patterns:
            matches = self._match_type.match(pattern, prompt)
            if matches is None or len(matches) == 0:
                continue

            if self._is_blocked:
                LOGGER.warning("Pattern was detected in the text", pattern=pattern)

                if self._redact:
                    for match in matches:
                        text_replace_builder.replace_text_get_insertion_index(
                            "[REDACTED]",
                            match.start(),
                            match.end(),
                        )

                return text_replace_builder.output_text, False, 1.0

            LOGGER.debug("Pattern matched the text", pattern=pattern)
            return text_replace_builder.output_text, True, -1.0

        if self._is_blocked:
            LOGGER.debug("None of the patterns were found in the text")
            return text_replace_builder.output_text, True, -1.0

        LOGGER.warning("None of the patterns matched the text")
        return text_replace_builder.output_text, False, 1.0

    def analyze_spans(self, prompt: str) -> list[dict]:
        """Return the character spans of the prompt that matched the patterns.

        Regex matching already yields exact offsets, so there is no attribution
        step. Only meaningful for a block-list scanner (is_blocked=True): there
        a match is the violation. For an allow-list scanner (is_blocked=False) a
        violation means *none* of the patterns matched, so there is no offending
        span to return and this returns an empty list.

        Returns a list of {"text", "score", "start", "end", "label"} dicts,
        where "label" is the pattern that matched.
        """
        if not self._is_blocked:
            return []

        spans: list[dict] = []
        for pattern in self._patterns:
            for match in self._match_type.match(pattern, prompt):
                start, end = match.start(), match.end()
                if end <= start:
                    continue
                spans.append(
                    {
                        "text": prompt[start:end],
                        "score": 1.0,
                        "start": start,
                        "end": end,
                        "label": pattern.pattern,
                    }
                )

        spans.sort(key=lambda s: s["start"])
        return spans
