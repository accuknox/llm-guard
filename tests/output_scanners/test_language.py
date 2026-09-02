import pytest

from llm_guard.output_scanners.language import Language


@pytest.mark.parametrize(
    "output,expected_valid,expected_score",
    [
        ("Just an output", True, -1.0),  # Correct output
        (
            "Me llamo Sofia. ¿Cómo te llamas?",
            False,
            1.0,
        ),  # Spanish
    ],
)
def test_scan(output, expected_valid, expected_score):
    scanner = Language(valid_languages=["en"])
    sanitized_output, valid, score = scanner.scan(
        "",
        output,
    )
    assert sanitized_output == output
    assert valid == expected_valid
    assert score == expected_score


@pytest.mark.parametrize(
    "output",
    [
        "Me llamo Sofia. ¿Cómo te llamas?",  # Spanish
        "Hello, my name is LLM Guard.",  # English
    ],
)
def test_scan_allow_select_all_allows_every_language(output):
    """allowSelectAll treats every language as valid, whatever valid_languages says."""
    scanner = Language(valid_languages=["en"], allowSelectAll=True)
    sanitized_output, valid, score = scanner.scan("", output)
    assert sanitized_output == output
    assert valid is True
    assert score == -1.0


def test_scan_allow_select_all_per_call_override():
    """allowSelectAll can be toggled per scan() call in both directions."""
    output = "Me llamo Sofia. ¿Cómo te llamas?"

    scanner = Language(valid_languages=["en"])
    assert scanner.scan("", output)[1] is False
    assert scanner.scan("", output, allowSelectAll=True)[1] is True

    scanner = Language(valid_languages=["en"], allowSelectAll=True)
    assert scanner.scan("", output)[1] is True
    assert scanner.scan("", output, allowSelectAll=False)[1] is False
