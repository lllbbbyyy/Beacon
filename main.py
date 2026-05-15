#!/usr/bin/env python3
"""CLI compatibility wrapper.

The implementation lives in ``chiplet_tuner/`` so the framework can be used as
a layered project instead of a single script.
"""

from chiplet_tuner.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
