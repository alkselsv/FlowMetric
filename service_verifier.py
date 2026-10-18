"""Compatibility alias for :mod:`flowmetric.service.astra`."""

import sys

from flowmetric.service import astra as _implementation

sys.modules[__name__] = _implementation
