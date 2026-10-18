"""Compatibility alias for :mod:`flowmetric.infrastructure.astra`."""

import sys

from flowmetric.infrastructure import astra as _implementation

sys.modules[__name__] = _implementation
