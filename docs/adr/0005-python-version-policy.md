# 0005: Supported Python versions

Status: accepted (2026-11-20)

## Context
The build request asked for Python 3.10 to 3.13. Python 3.10 reached end of life in October 2026, and the
current numpy (2.5) and scipy (1.18) require Python 3.12 or later.

## Decision
`requires-python = ">=3.12"`. CI tests 3.12, 3.13 and 3.14 on Linux, macOS and Windows. Local development
uses 3.12 (the oldest supported version) so that newer syntax is not used by accident. When a Python
version reaches end of life, it is dropped in the next minor release, following SPEC 0 from the
scientific Python community.

## Alternatives
Keep 3.10 and 3.11: CI on those versions would test old numpy and scipy, doubling the support surface for
little gain.

## Consequences
Extras are capped by upstream support (for example SB3 lists Python up to 3.13, PettingZoo requires
< 3.15); extras are tested only on Python versions their upstream supports.
