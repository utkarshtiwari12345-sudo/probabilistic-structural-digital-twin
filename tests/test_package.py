"""Smoke tests for the project package."""

import probdt


def test_package_version_is_defined() -> None:
    """The installed package exposes its declared initial version."""
    assert probdt.__version__ == "0.1.0"
