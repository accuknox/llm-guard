# Code Scanner

This scanner can be particularly useful in applications that need to reject code snippets in specific languages, or any code at all.

## Attack scenario

In some contexts, having a language model inadvertently produce code in its output might be deemed undesirable or risky.
For instance, a user might exploit the model to generate malicious scripts or probe it for potential vulnerabilities.
Controlling and inspecting the code in the model's output can be paramount in ensuring user safety and system integrity.

## How it works

Utilizing a multi-label code-language classifier, the scanner can identify code snippets across various programming
languages.

The `languages` given to the scanner are the ones that are **blocked**: code detected in any of them makes the
output invalid, while code in any other language passes through. Selecting **all** supported languages - which is
also what passing no languages at all means - blocks any code whatsoever, making the scanner behave like the
[BanCode](./ban_code.md) scanner.

The scanner extracts code snippets from Markdown and detects the following languages:

- AWK
- Bash
- Batch
- C
- C#
- C++
- Dockerfile
- Go
- Java
- JavaScript
- Kotlin
- Lua
- Makefile
- Perl
- PowerShell
- Python
- R
- Ruby
- Rust
- SQL
- Scala
- Swift
- Terraform
- jq

!!! note

    To ban all code snippets, either select every supported language or use the dedicated
    [BanCode](./ban_code.md) scanner.

## Usage

```python
from llm_guard.output_scanners import Code

scanner = Code(languages=["Python"])  # Python is blocked, other languages pass
sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
```

## Optimization Strategies

[Read more](../tutorials/optimization.md)

## Benchmarks

Test setup:

- Platform: Amazon Linux 2
- Python Version: 3.11.6
- Input length: 159
- Test times: 5

Run the following script:

```sh
python benchmarks/run.py output Code
```

Results:

| Instance                    | Latency Variance | Latency 90 Percentile | Latency 95 Percentile | Latency 99 Percentile | Average Latency (ms) | QPS      |
|-----------------------------|------------------|-----------------------|-----------------------|-----------------------|----------------------|----------|
| AWS m5.xlarge               | 2.76             | 109.82                | 136.05                | 157.04                | 57.16                | 2781.88  |
| AWS m5.xlarge with ONNX     | 0.03             | 26.15                 | 28.35                 | 30.11                 | 20.22                | 7864.68  |
| AWS g5.xlarge GPU           | 32.10            | 273.88                | 363.37                | 434.96                | 94.52                | 1682.22  |
| AWS g5.xlarge GPU with ONNX | 0.01             | 6.79                  | 8.32                  | 9.53                  | 3.73                 | 42667.01 |
