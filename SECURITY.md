# Security Policy

## Supported Versions

We follow [semantic versioning](https://semver.org/). The most recent major
release receives regular security updates. The previous major release continues
to receive critical fixes for at least six months after a new major version is
published.

| Version | Supported |
| ------- | --------- |
| 2.x     | :white_check_mark: |
| 1.x     | :white_check_mark: |
| < 1.0   | :x: |

## Reporting a Vulnerability

If you discover a security issue, please email **security@legalai.com** with a
thorough description and, if possible, steps to reproduce. We strive to respond
within two business days. Please refrain from publicly disclosing the issue
until we have confirmed the vulnerability and prepared a fix or advisory.

## Security Update Process

1. Reports are triaged by the Legal AI Team.
2. Valid issues are fixed in a private branch and patch releases are created.
3. After releasing a fix, we publish a security advisory in this repository and
   request a CVE when appropriate.

## Recommendations

To keep your deployment secure:

- Use the latest supported version of Python (3.9 or later).
- Keep dependencies up to date via `pip install -U -r requirements.txt`.
- Store secrets in environment variables or a dedicated secrets manager.

Thank you for helping us keep the Legal AI System safe!
