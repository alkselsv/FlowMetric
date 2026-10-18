#!/usr/bin/env python3
"""Compatibility entry point for :mod:`flowmetric.pipeline`."""

from __future__ import annotations

import sys

from flowmetric import pipeline as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
