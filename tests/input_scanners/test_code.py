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
