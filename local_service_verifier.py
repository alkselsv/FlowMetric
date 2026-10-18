"""Compatibility alias for :mod:`flowmetric.service.local`."""

import sys

from flowmetric.service import local as _implementation

sys.modules[__name__] = _implementation
