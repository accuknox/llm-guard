import pytest

from llm_guard.output_scanners.sensitive import Sensitive


@pytest.mark.parametrize(
    "output,expected_output,expected_valid,expected_score",
    [
        (
            "Sorry, I can't",
            "Sorry, I can't",
            True,
            -1.0,
        ),  # No sensitive information
        (
            "john.doe@protectai.com is my email",
            "<EMAIL_ADDRESS> is my email",
            False,
            1.0,
        ),  # Name in the output
    ],
)
def test_scan(output, expected_output, expected_valid, expected_score):
    scanner = Sensitive(redact=True, entity_types=["EMAIL_ADDRESS", "EMAIL_ADDRESS_RE"])
    sanitized_output, valid, score = scanner.scan("", output)
    assert sanitized_output == expected_output
    assert valid == expected_valid
    assert score == expected_score


def test_scan_allow_select_all_detects_every_entity_type():
    """allowSelectAll looks for every entity the analyzer knows, whatever entity_types says."""
    output = "The IP address is 192.168.1.100."

    # IP_ADDRESS is outside the configured entity types, so it is left alone.
    scanner = Sensitive(redact=True, entity_types=["EMAIL_ADDRESS"])
    sanitized_output, valid, _ = scanner.scan("", output)
    assert sanitized_output == output
    assert valid is True

    scanner = Sensitive(redact=True, entity_types=["EMAIL_ADDRESS"], allowSelectAll=True)
    sanitized_output, valid, _ = scanner.scan("", output)
    assert "192.168.1.100" not in sanitized_output
    assert valid is False


def test_scan_allow_select_all_per_call_override():
    """allowSelectAll can be toggled per scan() call in both directions."""
    output = "The IP address is 192.168.1.100."

    scanner = Sensitive(entity_types=["EMAIL_ADDRESS"])
    assert scanner.scan("", output)[1] is True
    assert scanner.scan("", output, allowSelectAll=True)[1] is False

    scanner = Sensitive(entity_types=["EMAIL_ADDRESS"], allowSelectAll=True)
    assert scanner.scan("", output)[1] is False
    assert scanner.scan("", output, allowSelectAll=False)[1] is True
