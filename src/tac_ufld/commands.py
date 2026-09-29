"""Sub-commands beyond run / check-data / datasets / validate-dataset.
Each is registered here and implemented in its own module; heavy imports
happen inside the handlers so ``--help`` stays fast."""

from __future__ import annotations

import argparse

HANDLERS: dict = {}


def register(sub: argparse._SubParsersAction) -> None:
    """Add every extra sub-command to the CLI parser."""
