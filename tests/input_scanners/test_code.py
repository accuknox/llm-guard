import pytest

from llm_guard.input_scanners.code import SUPPORTED_LANGUAGES, Code


@pytest.mark.parametrize(
    "prompt,expected_valid,expected_score",
    [
        ("Just a simple prompt", True, -1.0),  # Simple prompt
        (
            """Fix the following JS code:

```js
function sayHello() {
    return "Hello, World!";
}
console.log(sayHello());
```""",
            True,
            -1.0,
        ),  # JavaScript is allowed
        (
            "Is this correct way to make function:\n```\nimport os\ndef new_func():\n```",
            False,
            1.0,
        ),  # Python is not allowed
        (
            """function sayHello() {
    return "Hello, World!";
}
console.log(sayHello());""",
            True,
            -1.0,
        ),
    ],
)
def test_scan(prompt, expected_valid, expected_score):
    scanner = Code(languages=["JavaScript"])
    sanitized_prompt, valid, score = scanner.scan(prompt)
    assert sanitized_prompt == prompt
    assert valid == expected_valid
    assert score == expected_score


@pytest.mark.parametrize(
    "languages",
    [
        None,  # No languages means nothing is allowed
        [],  # Empty selection, same thing
    ],
)
@pytest.mark.parametrize(
    "prompt,expected_valid,expected_score",
    [
        ("Just a simple prompt", True, -1.0),  # No code at all
        ("Is this correct way to make function:\n```\nimport os\ndef new_func():\n```", False, 1.0),
        ("""```js
function sayHello() {
    return "Hello, World!";
}
```""", False, 1.0),
    ],
)
def test_scan_no_languages_bans_all_code(languages, prompt, expected_valid, expected_score):
    """Selecting no languages makes Code behave like the BanCode scanner."""
    scanner = Code(languages=languages)
    sanitized_prompt, valid, score = scanner.scan(prompt)
    assert sanitized_prompt == prompt
    assert valid == expected_valid
    assert score == expected_score


@pytest.mark.parametrize(
    "kwargs",
    [
        {"languages": SUPPORTED_LANGUAGES},  # Every language selected explicitly
        {"languages": ["JavaScript"], "allowSelectAll": True},  # allowSelectAll wins
        {"allowSelectAll": True},
    ],
)
@pytest.mark.parametrize(
    "prompt",
    [
        "Just a simple prompt",
        "Is this correct way to make function:\n```\nimport os\ndef new_func():\n```",
        """```js
function sayHello() {
    return "Hello, World!";
}
```""",
    ],
)
def test_scan_all_languages_allows_all_code(kwargs, prompt):
    """Selecting every language, or allowSelectAll, allows any code."""
    scanner = Code(**kwargs)
    sanitized_prompt, valid, score = scanner.scan(prompt)
    assert sanitized_prompt == prompt
    assert valid is True
    assert score == -1.0


def test_scan_per_call_overrides():
    """languages and allowSelectAll can be overridden per scan() call."""
    prompt = "Is this correct way to make function:\n```\nimport os\ndef new_func():\n```"

    scanner = Code(languages=["JavaScript"])
    assert scanner.scan(prompt)[1] is False
    assert scanner.scan(prompt, allowSelectAll=True)[1] is True
    assert scanner.scan(prompt, languages=["Python"])[1] is True

    scanner = Code(languages=["Python"], allowSelectAll=True)
    assert scanner.scan(prompt)[1] is True
    assert scanner.scan(prompt, allowSelectAll=False)[1] is True
    # An explicit empty list allows nothing, it does not fall back to the default
    assert scanner.scan(prompt, languages=[], allowSelectAll=False)[1] is False
