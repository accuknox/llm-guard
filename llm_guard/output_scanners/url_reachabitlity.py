from __future__ import annotations

import requests

from llm_guard.util import extract_urls, get_logger

from .base import Scanner

LOGGER = get_logger()


class URLReachability(Scanner):
    """
    This scanner checks URLs for their reachability.
    """

    def __init__(self, *, success_status_codes: list[int] | None = None, timeout: int = 5) -> None:
        """
        Parameters:
            success_status_codes: A list of status codes that are considered as successful.
            timeout: The timeout in seconds for the HTTP requests.
        """
        if success_status_codes is None:
            success_status_codes = [
                requests.codes.ok,
                requests.codes.created,
                requests.codes.accepted,
            ]

        self._success_status_codes = success_status_codes
        self._timeout = timeout

    def is_reachable(
        self,
        url: str,
        success_status_codes: list[int] | None = None,
        timeout: int | None = None,
    ) -> bool:
        """
        Check if the URL is reachable.
        """
        if success_status_codes is None:
            success_status_codes = self._success_status_codes
        if timeout is None:
            timeout = self._timeout

        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code in success_status_codes
        except requests.RequestException:
            return False

    def scan(
        self,
        prompt: str,
        output: str,
        success_status_codes: list[int] | None = None,
        timeout: int | None = None,
    ) -> tuple[str, bool, float]:
        urls = extract_urls(output)
        if not urls:
            return output, True, -1.0

        LOGGER.debug("Found URLs in the output", len=len(urls))

        unreachable_urls = [
            url
            for url in urls
            if not self.is_reachable(
                url, success_status_codes=success_status_codes, timeout=timeout
            )
        ]

        if unreachable_urls:
            LOGGER.warning("Unreachable URLs detected", urls=unreachable_urls)
            return output, False, 1.0

        LOGGER.debug("All URLs are reachable.")
        return output, True, -1.0
