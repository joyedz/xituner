"""Backward-compatible shim. Use `scripts.compare --use-case brand_voice` instead.

This script used to be the comparison orchestrator, with one use case hardcoded
into it. It now forwards to the generic `scripts/compare.py`.

The shim exists rather than a rename because a Colab notebook already open
against this path would break mid-session otherwise, and breaking somebody's
running experiment for the sake of a tidier filename is a bad trade.
"""

from __future__ import annotations

import sys

from scripts.compare import main as compare_main


def main() -> None:
    argv = sys.argv[1:]
    if not any(a == "--use-case" or a.startswith("--use-case=") for a in argv):
        argv = ["--use-case", "brand_voice", *argv]
    print(
        "note: compare_voice.py is a shim. The generic entry point is\n"
        "      python -m scripts.compare --use-case brand_voice ...\n"
    )
    compare_main(argv)


if __name__ == "__main__":
    main()
