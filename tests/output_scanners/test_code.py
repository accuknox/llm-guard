import pytest

from llm_guard.output_scanners.code import Code


@pytest.mark.parametrize(
    "output,expected_valid,expected_score",
    [
        ("", True, -1.0),  # Empty prompt
        (
            """Sure! Here's a simple JavaScript function that returns "Hello, World!":

```js
function sayHello() {
    return "Hello, World!";
}
console.log(sayHello());
```""",
            False,
            1.0,
        ),  # Blocked language
        (
            """Sure, below you can find the code:

```go
package main

import "fmt"

func main() {
    text := "hello world"
    fmt.Println(text)
}
```""",
            True,
            -1.0,
        ),  # Language that is not blocked
        (
            """You can achieve this using Python's set data structure. Let's say you have a list of strings named `list_of_strings` and another list named `available_strings`. You can check if all elements of `list_of_strings` exist in `available_strings` by converting both lists to sets and then using the set's `issubset` method.

Here's a simple example:

```python
list_of_strings = ["apple", "banana", "cherry"]
available_strings = ["apple", "banana", "cherry", "date", "fig"]

# Convert lists to sets
set_of_strings = set(list_of_strings)
set_of_available = set(available_strings)

# Check if all elements of list_of_strings exist in available_strings
all_exist = set_of_strings.issubset(set_of_available)

print(all_exist)  # This will print True if all elements exist, False otherwise
```

This approach takes advantage of the set's properties to efficiently check for membership.""",
            False,
            1.0,
        ),  # Long output in Python
    ],
)
def test_scan(output, expected_valid, expected_score):
    scanner = Code(languages=["JavaScript", "Python"])
    sanitized_output, valid, score = scanner.scan("", output)
    assert sanitized_output == output
    assert valid == expected_valid
    assert score == expected_score


@pytest.mark.parametrize(
    "output,expected_valid,expected_score",
    [
        ("", True, -1.0),  # Empty output
        ("Just a simple output", True, -1.0),  # No code at all
        (
            """Sure, below you can find the code:

```go
package main

import "fmt"

func main() {
    text := "hello world"
    fmt.Println(text)
}
```""",
            False,
            1.0,
        ),  # Go is flagged even though only JavaScript is configured
    ],
)
def test_scan_allow_select_all_bans_all_code(output, expected_valid, expected_score):
    """allowSelectAll blocks every language, whatever `languages` says."""
    scanner = Code(languages=["JavaScript"], allowSelectAll=True)
    sanitized_output, valid, score = scanner.scan("", output)
    assert sanitized_output == output
    assert valid == expected_valid
    assert score == expected_score


def test_scan_allow_select_all_per_call_override():
    """allowSelectAll can be toggled per scan() call in both directions."""
    output = """```go
package main

func main() {}
```"""

    scanner = Code(languages=["JavaScript"])
    assert scanner.scan("", output)[1] is True
    assert scanner.scan("", output, allowSelectAll=True)[1] is False

    scanner = Code(languages=["JavaScript"], allowSelectAll=True)
    assert scanner.scan("", output)[1] is False
    assert scanner.scan("", output, allowSelectAll=False)[1] is True
