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
            False,
            1.0,
        ),  # JavaScript is blocked
        (
            "Is this correct way to make function:\n```\nimport os\ndef new_func():\n```",
            True,
            -1.0,
        ),  # Python is not blocked
        (
            """function sayHello() {
    return "Hello, World!";
}
console.log(sayHello());""",
            False,
            1.0,
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
        None,  # No languages means every language
        SUPPORTED_LANGUAGES,  # Every language selected explicitly
    ],
)
@pytest.mark.parametrize(
    "prompt,expected_valid,expected_score",
    [
        ("Just a simple prompt", True, -1.0),  # No code at all
        (
            "Is this correct way to make function:\n```\nimport os\ndef new_func():\n```",
            False,
            1.0,
        ),  # Python is blocked too when every language is selected
        (
            """```js
function sayHello() {
    return "Hello, World!";
}
```""",
            False,
            1.0,
        ),
    ],
)
def test_scan_all_languages_bans_all_code(languages, prompt, expected_valid, expected_score):
    """Selecting every language makes Code behave like the BanCode scanner."""
    scanner = Code(languages=languages)
    sanitized_prompt, valid, score = scanner.scan(prompt)
    assert sanitized_prompt == prompt
    assert valid == expected_valid
    assert score == expected_score


@pytest.mark.parametrize(
    "prompt,expected_valid,expected_score",
    [
        ("Just a simple prompt", True, -1.0),  # No code at all
        (
            "Is this correct way to make function:\n```\nimport os\ndef new_func():\n```",
            False,
            1.0,
        ),  # Python is flagged even though only JavaScript is configured
        (
            """```js
function sayHello() {
    return "Hello, World!";
}
```""",
            False,
            1.0,
        ),
    ],
)
def test_scan_allow_select_all_bans_all_code(prompt, expected_valid, expected_score):
    """allowSelectAll blocks every language, whatever `languages` says."""
    scanner = Code(languages=["JavaScript"], allowSelectAll=True)
    sanitized_prompt, valid, score = scanner.scan(prompt)
    assert sanitized_prompt == prompt
    assert valid == expected_valid
    assert score == expected_score


def test_scan_allow_select_all_per_call_override():
    """allowSelectAll can be toggled per scan() call in both directions."""
    prompt = "Is this correct way to make function:\n```\nimport os\ndef new_func():\n```"

    scanner = Code(languages=["JavaScript"])
    assert scanner.scan(prompt)[1] is True
    assert scanner.scan(prompt, allowSelectAll=True)[1] is False

    scanner = Code(languages=["JavaScript"], allowSelectAll=True)
    assert scanner.scan(prompt)[1] is False
    assert scanner.scan(prompt, allowSelectAll=False)[1] is True
