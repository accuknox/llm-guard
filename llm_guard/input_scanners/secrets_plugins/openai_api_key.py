"""
This plugin searches for OpenAI API Keys.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class OpenAIApiKeyDetector(RegexBasedDetector):
    """Scans for OpenAI API Keys."""

    @property
    def secret_type(self) -> str:
        return "OpenAI API Key"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            re.compile(
                r"""\b(?:sk-proj-[A-Za-z0-9_-]{20,200}|sk-[A-Za-z0-9_-]{20,200}|sv-[A-Za-z0-9_-]{20,200})\b"""
            )
        ]
