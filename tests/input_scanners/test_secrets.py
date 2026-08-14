from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_guard.input_scanners.secrets import Secrets


@pytest.mark.parametrize(
    "prompt,expected_prompt,expected_valid,expected_score",
    [
        (
            "Just a simple prompt",
            "Just a simple prompt",
            True,
            -1.0,
        ),  # Prompt without sensitive data
        (
            'I need to pass a key\naws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"',  # gitleaks:allow
            'I need to pass a key\naws_secret_access_key="************"',
            False,
            1.0,
        ),  # Prompt with key
        (
            "My github token is: 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",  # gitleaks:allow
            "My github token is: ************",
            False,
            1.0,
        ),  # Prompt with Github token
        (
            "My JWT token is: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",  # gitleaks:allow
            "My JWT token is: ******",
            False,
            1.0,
        ),  # Prompt with a JWT token
        (
            "Use this URL: https://username:password@llm-guard.com",
            "Use this URL: https://username:******@llm-guard.com",
            False,
            1.0,
        ),  # Prompt with HTTP basic auth
        (
            "Securely and attractively display eyewear, allow easy customer browsing,Lack of secure and appealing eyewear displays,Custom sunglass display, allow easy customer browsing,Lack of secure and appealing eyewear displays,Custom sunglass displa",
            "Securely and attractively display eyewear, allow easy customer browsing,Lack of secure and appealing eyewear displays,Custom sunglass display, allow easy customer browsing,Lack of secure and appealing eyewear displays,Custom sunglass displa",
            True,
            -1.0,
        ),  # False-positive
    ],
)
def test_scan(prompt, expected_prompt, expected_valid, expected_score):
    scanner = Secrets()
    sanitized_prompt, valid, score = scanner.scan(prompt)
    print(sanitized_prompt)

    assert sanitized_prompt == expected_prompt
    assert valid == expected_valid
    assert score == expected_score


def test_scan_is_thread_safe():
    """Concurrent scans must not lose detections.

    detect_secrets keeps its plugin configuration in a process-global singleton
    and transient_settings() swaps it in and out around every scan, so parallel
    scans corrupt each other's plugin set: the scan either raises KeyError on a
    plugin name, or silently reports the text as safe. Both are unacceptable in
    a security scanner, and the second fails open. Secrets.scan serializes entry
    into that context; without it this test loses detections and often raises.
    """
    scanner = Secrets()
    prompt_with_secret = "My github token is: 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"  # gitleaks:allow

    def scan(index):
        has_secret = bool(index % 2)
        prompt = prompt_with_secret if has_secret else "Just a simple prompt"
        _, valid, _ = scanner.scan(prompt)
        return has_secret, valid

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(scan, range(60)))

    missed = [has_secret for has_secret, valid in results if has_secret and valid]
    false_positives = [valid for has_secret, valid in results if not has_secret and not valid]

    assert not missed, f"{len(missed)} of 30 secrets missed under concurrency"
    assert not false_positives, f"{len(false_positives)} of 30 clean prompts flagged"
